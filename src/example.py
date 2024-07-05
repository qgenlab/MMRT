import pickle
import argparse
from MMRT import MMRT

def parse_cl():
    parser = argparse.ArgumentParser(
        description="Train, test, and generate predictions using MMRT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--train_data",
        default = None,
        type = str,
        nargs = '*',
        help = "Path to training data (can be multiple)."
    )
    
    parser.add_argument(
        "--test_data",
        default = None,
        type = str,
        nargs = '*',
        help = "Path to testing data (can be multiple)."
    )
     
    parser.add_argument(
        "--train_ratio",
        default = 0.9,
        type = float,
        help = "Train ratio when splitting training data into train-test split (used only if train_data is None)."
    )
    
    parser.add_argument(
        "--batch_size",
        default = 32,
        type = int,
        help = "Batch size for training model."
    )
     
    parser.add_argument(
        "--save_log",
        action=argparse.BooleanOptionalAction,
        help = "Whether to save log of performance during training."
    )
     
    parser.add_argument(
        "--resume_log",
        default = False,
        action='store_true',
        help = "Whether to overwrite existing log or resume (usually True if checkpoint is used)."
    )
      
    parser.add_argument(
        "--save_model",
        default = False,
        action='store_true',
        help = "Whether to save model/optimizer checkpoint."
    )
       
    parser.add_argument(
        "--save_path",
        default = './',
        type = str,
        help = "Path to save log, models, predictions folders."
    )
           
    parser.add_argument(
        "--model_name",
        default = 'MMRT',
        type = str,
        help = "Name for model (used for output file naming)."
    )
    
    parser.add_argument(
        "--device",
        type = str,
        default = None,
        help = "The device to use."
    )
        
    parser.add_argument(
        "--saved_model",
        type = str,
        default = None,
        help = "The path to saved model/model checkpoint."
    )
            
    parser.add_argument(
        "--epochs",
        type = int,
        default = 100,
        help = "Number of epochs to train model."
    )
               
    parser.add_argument(
        "--learning_rate",
        type = float,
        default = 1e-5,
        help = "Learning rate to use during training."
    )
                  
    parser.add_argument(
        "--random_seed",
        type = int,
        default = None,
        help = "Random seed used during training (use for reproducibility)."
    )
                     
    parser.add_argument(
        "--cadence",
        type = int,
        default = 40,
        help = "Number of epochs between writing to log and/or saving checkpoints."
    )
                        
    parser.add_argument(
        "--save_prediction",
        default = False,
        action='store_true',
        help = "Whether to save final predictions for training data."
    )
   
    return parser.parse_args()


def main():
    """
    Usage:
    python example.py \
    --train_data /mnt/labshare/bryceForrest/esm_vectors/window_2/parEparD_Laub2015_all_win_1_1.p \
    /mnt/labshare/bryceForrest/esm_vectors/window_2/parEparD_Laub2015_all_win_1_2.p \
    --test_data /mnt/labshare/bryceForrest/esm_vectors/window_2/parEparD_Laub2015_all_win_1_3.p \
    /mnt/labshare/bryceForrest/esm_vectors/window_2/parEparD_Laub2015_all_win_1_4.p \
    --batch_size 32 \
    --save_log \
    --save_model \
    --save_path './' \
    --model_name 'parEparD_Laub2015_all_win_1' \
    --device 4 \
    --epochs 10 \
    --learning_rate 1e-5 \
    --random_seed 0 \
    --cadence 2 \
    --save_prediction    
    """
    args = parse_cl()
    
    if args.train_data is not None:
        train_data = []
        for d in args.train_data:
            train_data.append(pickle.load(open(d, 'rb')))
    else:
        train_data = None
    
    if args.test_data is not None:
        test_data = []
        for d in args.test_data:
            test_data.append(pickle.load(open(d, 'rb')))
    else:
        test_data = None
    
    mmrt = MMRT(train_data=train_data,
            test_data=test_data,
            train_ratio=args.train_ratio,
            batch_size=args.batch_size,
            save_log=args.save_log,
            resume_log=args.resume_log,
            save_model=args.save_model,
            save_path=args.save_path,
            model_name=args.model_name,
            device=args.device)
    
    if args.saved_model is not None:
        mmrt.load_checkpoint(args.saved_model)
    else:
        mmrt.train(epochs=args.epochs,
              learning_rate=args.learning_rate,
              random_seed=args.random_seed,
              cadence=args.cadence)
    
    mmrt.test(args.save_prediction)
    
if __name__=='__main__':
    main()