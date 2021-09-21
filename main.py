import os
import argparse
import csv
import numpy as np
from sampler import Dataset
from model import ConvRec
from tqdm import tqdm
import pickle
from util import *
import zipfile
import torch
import torch.optim as optim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='last_fm')
    parser.add_argument('--top_k', default=10, type=int)
    
    parser.add_argument('--train_dir', default='default')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--maxlen', default=30, type=int)

    parser.add_argument('--embed_dim', default=200, type=int) 
    parser.add_argument('--ffn_embed_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.2, type=float) 
    parser.add_argument('--weight_dropout', default=0.2, type=float)


    parser.add_argument('--layers', default=2, type=int) 
    parser.add_argument('--heads', default=1, type=int) 

    parser.add_argument('--decoder_kernel_size_list', default = [5, 5]) #depends on the number of layer
    
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--num_neg_samples', default = 400, type=int) #Note: 100 is sufficient
    parser.add_argument('--eval_epoch', default = 5, type=int)
    parser.add_argument('--testing', default=False, action='store_true')
    parser.add_argument('--test_data', nargs='+')
    
    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")
    
    parser.add_argument('--computing_device', default=computing_device)

    # # Get the arguments
    try:
        #if running from command line
        args = parser.parse_args()
    except:
        #if running in IDEs
        args = parser.parse_known_args()[0] 

    
    result_path = 'results/'+args.dataset + '_' + args.train_dir
    

    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    with open(os.path.join(result_path, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    
    if not args.testing:
        if os.path.exists("data/"+args.dataset + '.pkl'):
            pickle_in = open("data/"+args.dataset+".pkl","rb")
            dataset = pickle.load(pickle_in)
        else:
            dataset = data_partition(args.dataset)
            pickle_out = open("data/"+args.dataset+".pkl","wb")
            pickle.dump(dataset, pickle_out)
            pickle_out.close()
    else:
        if os.path.exists(os.path.join("data/test_data", args.dataset + '.pkl.zip')):
            with zipfile.ZipFile(os.path.join("data/test_data", args.dataset + '.pkl.zip'), 'r') as zip_ref:
                zip_ref.extractall("data/test_data")

            pickle_in = open(os.path.join("data/test_data", args.dataset + '.pkl'),"rb")
            dataset = pickle.load(pickle_in)
            print("Length of dataset is: ",len(dataset))
        else:
            dataset = data_partition(args.dataset)
            pickle_out = open("data/test_data"+args.dataset+".pkl","wb")
            pickle.dump(dataset, pickle_out)
            pickle_out.close()
    
    # print(dataset)
    
    if not args.testing:
        [train, valid, test, itemnum] = dataset
    else:
        [train, valid, itemnum] = dataset
    
    print("Number of sessions:",len(train)+len(valid)+(len(test) if not args.testing else 0))
    print("Number of items:", itemnum)

    action = 0
    for i in train:
        action += np.count_nonzero(i)
    
    for i in valid:
        action += np.count_nonzero(i)
    
    if not args.testing:
        for i in test:
            action += np.count_nonzero(i)

    print("Number of actions:", action)
    
    print("Average length of sessions:", action/(len(train)+len(valid)+(len(test) if not args.testing else 0)))


    num_batch = len(train) // args.batch_size
    print("The batch size is:", num_batch)
    
    f = open(os.path.join(result_path, 'log.txt'), 'w')
    
    conv_model = ConvRec(args, itemnum)
    conv_model = conv_model.to(args.computing_device, non_blocking=True)
    
    # Note: testing a pretrained model
    # Removing this module for bias testing
    if os.path.exists(result_path+"pretrained_model.pth") and not args.testing:
        conv_model.load_state_dict(torch.load(result_path+"pretrained_model.pth"))       
        t_test = evaluate(conv_model, test, itemnum, args, num_workers=4)
        model_performance = "Model performance on test: "+str(t_test)
        print(model_performance)


    optimizer = optim.Adam(conv_model.parameters(), lr = args.lr, betas=(0.9, 0.98), weight_decay = 0.0)

    f.write(str(args)+'\n')
    f.flush()
    

    best_val_loss = 1e6
    train_losses = []
    val_losses = []

    best_ndcg = 0
    best_hit = 0
    model_performance = None

    stop_count = 0
    total_epochs = 1    
    
    dataset = Dataset(train, args, itemnum, train=True)

     
    sampler = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    for epoch in range(1, args.num_epochs + 1):  
        conv_model.train()

        epoch_losses = []

        
        for step, (seq, pos) in tqdm(enumerate(sampler), total=len(sampler)):  
                
            optimizer.zero_grad()


            seq = torch.LongTensor(seq).to(args.computing_device, non_blocking=True)
            pos = torch.LongTensor(pos).to(args.computing_device, non_blocking=True)

            loss, _  = conv_model.forward(seq, pos=pos)

            epoch_losses.append(loss.item())

            # Compute gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            
        
        if total_epochs % args.eval_epoch == 0:

            t_valid = evaluate(conv_model, valid, itemnum, args, num_workers=4)
            
            print ('\nnum of steps:%d, valid (MRR@%d: %.4f, NDCG@%d: %.4f, HR@%d: %.4f), valid (MRR@%d: %.4f, NDCG@%d: %.4f, HR@%d: %.4f)' % (total_epochs, args.top_k, t_valid[0], args.top_k, t_valid[1], args.top_k, t_valid[2],
            args.top_k+10, t_valid[3], args.top_k+10, t_valid[4], args.top_k+10, t_valid[5]))

            f.write(str(t_valid) + '\n')
            f.flush()

            
            if t_valid[0]>best_ndcg:
                best_ndcg = t_valid[0]
                torch.save(conv_model.state_dict(), result_path+"pretrained_model.pth")
                stop_count = 1
            else:
                stop_count += 1

            if stop_count == 3: #model did not improve 3 consequetive times
                break
                
        total_epochs += 1

        train_loss = np.mean(epoch_losses)
        print(str(epoch) + "epoch loss", train_loss)


    conv_model = ConvRec(args, itemnum)
    conv_model.load_state_dict(torch.load(result_path+"pretrained_model.pth"))
    
    conv_model = conv_model.to(args.computing_device)
        
    if args.testing:
        # test set can either be 'clicks' or 'buys'
        for test_set in args.test_data:
            with open(os.path.join('data/test_data', args.dataset+'_'+test_set+'.pkl'), 'rb') as test_file:
                test = pickle.load(test_file)
            t_test = evaluate(conv_model, test, itemnum, args, num_workers=4)

            model_performance = "Model performance on " + test_set + " test: "+str(t_test)
            print(model_performance)
            
            f.write(model_performance+'\n\n')
            f.flush()
    else:
        t_test = evaluate(conv_model, test, itemnum, args, num_workers=4)

        model_performance = "Model performance on test: "+str(t_test)
        print(model_performance)

        f.write(model_performance+'\n')
        f.flush()

    f.close()

    print("Done")
