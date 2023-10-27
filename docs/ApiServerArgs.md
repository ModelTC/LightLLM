### api_server args 

#### --host
default is "127.0.0.1",  
http server host ip.

#### --port
default is "8000",  
http server port.

#### --tp
default is 1,  
tensor parrall size.

#### --model_dir

the model weight dir path, the server will load config, weights and tokenizer from this dir.

#### --tokenizer_mode
default is "slow",  
tokenizer load mode, which can be "slow" or "auto", "slow" mode always loads fast but runs slow, "slow" mode is good for debugging and testing, when you want to get best performance, please use "auto" mode.

#### --max_total_token_num

default is 6000,  
the total token num the gpu and model can support, a sample about how to set this arg:   
gpu: use 2 A100 80G, (--tp 2)  
model: llama-7b,  
dtype: fp16,  
llama-7b hidden_size is 4096, layers num is 32,   
the gpu mem left after gpu load all weights,   

80 * 2 - 7 * 2 = 146G  

gpu mem for one Token kv cache:   

4096 * 2 * 2 * 32 / 1024 / 1024 / 1024 =  0.000488281G  

the max token num:    

146 / 0.000488281 ≈ 299008  

Of course, this value cannot be directly set, because extra gpu mem will be used during the model inference，We need to multiply this value by a ratio:  

max_total_token_num = int(299008 * ratio)   

We recommend setting the ratio between 0.8 and 0.9, perhaps slightly higher. if OOM error happens, you can reduce the ratio or arg "batch_max_tokens".  

#### --batch_max_tokens

the server will merge requests in waiting list to a batch to inference first (called prefill), batch_max_tokens will control tokens num of the merged batch, Reasonable setting of this arg can prevent OOM. if not set, it will be 1 / 6 * max_total_token_num.

#### --eos_id

defautl is 2,  
eos_id refers to the token id used to indicate the end of a sequence.

#### --running_max_req_size  

default is 1000,   
the max size for running requests in the same time.  

#### --max_req_input_len
default is 2048,  
the max value for one reqest's input token len.  


#### --max_req_total_len
default is 3072,  
the max value for req_input_len + req_output_len.
