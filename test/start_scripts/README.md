# LightLLM DeepSeek Model Startup Scripts

This directory contains various startup scripts for deploying DeepSeek models with LightLLM, covering different deployment modes and hardware configurations.

## Script Categories

### Single Node Deployment Scripts

- `single_node_tp.sh` - Single node tensor parallelism (TP) mode
- `single_node_ep.sh` - Single node expert parallelism (EP) mode

### Multi-Node Deployment Scripts

- `multi_node_tp_node0.sh` - Multi-node tensor parallelism node 0
- `multi_node_tp_node1.sh` - Multi-node tensor parallelism node 1
- `multi_node_ep_node0.sh` - Multi-node expert parallelism node 0
- `multi_node_ep_node1.sh` - Multi-node expert parallelism node 1

### PD Separated Deployment Scripts

#### Single PD Master Mode
- `single_pd_master/pd_master.sh` - PD Master service
- `single_pd_master/pd_prefill.sh` - Prefill service
- `single_pd_master/pd_decode.sh` - Decode service

#### Multi PD Master Mode
- `multi_pd_master/config_server.sh` - Configuration server
- `multi_pd_master/pd_master_1.sh` - PD Master 1
- `multi_pd_master/pd_master_2.sh` - PD Master 2
- `multi_pd_master/pd_prefill.sh` - Prefill service
- `multi_pd_master/pd_decode.sh` - Decode service

## Usage Instructions

### 1. Single Node TP Mode

```bash
# Modify model path and run directly
sh single_node_tp.sh
```

### 2. Single Node EP Mode

```bash
# Modify model path and run directly
sh single_node_ep.sh
```

### 3. Multi-Node TP Mode

```bash
# Run on node 0
sh multi_node_tp_node0.sh <master_ip>

# Run on node 1
sh multi_node_tp_node1.sh <master_ip>
```

### 4. Multi-Node EP Mode

```bash
# Run on node 0
sh multi_node_ep_node0.sh <master_ip>

# Run on node 1
sh multi_node_ep_node1.sh <master_ip>
```

### 5. Single PD Master Mode

```bash
# Step 1: Start PD Master
sh single_pd_master/pd_master.sh <pd_master_ip>

# Step 2: Start Prefill service
sh single_pd_master/pd_prefill.sh <host_ip> <pd_master_ip>

# Step 3: Start Decode service
sh single_pd_master/pd_decode.sh <host_ip> <pd_master_ip>
```

### 6. Multi PD Master Mode

```bash
# Step 1: Start configuration server
sh multi_pd_master/config_server.sh <config_server_host>

# Step 2: Start multiple PD Masters
sh multi_pd_master/pd_master_1.sh <host> <config_server_host>
sh multi_pd_master/pd_master_2.sh <host> <config_server_host>

# Step 3: Start Prefill and Decode services
sh multi_pd_master/pd_prefill.sh <host> <config_server_host>
sh multi_pd_master/pd_decode.sh <host> <config_server_host>
```

## Configuration Guide

### Environment Variables

- `LOADWORKER`: Model loading thread count, recommended 8-18
- `MOE_MODE`: Expert parallelism mode, set to EP to enable expert parallelism
- `DISABLE_KV_TRANS_USE_P2P`: Disable P2P communication optimization to transfer kv data
- `CUDA_VISIBLE_DEVICES`: Specify GPU devices to use

### Important Parameters

- `--model_dir`: Model file path
- `--tp`: Tensor parallelism degree
- `--dp`: Data parallelism degree
- `--nnodes`: Total number of nodes
- `--node_rank`: Current node rank
- `--nccl_host`: NCCL communication host address
- `--nccl_port`: NCCL communication port

## Hardware Configuration Recommendations

### H200 Single Node
- Recommended 8 GPUs, TP=8
- Memory: At least 128GB system memory

### H100 Dual Node
- Recommended 16 GPUs, TP=16
- Network: High bandwidth, low latency network connection

### General Recommendations
- Ensure GPU drivers and CUDA versions are compatible
- Check network connectivity and firewall settings
- Monitor GPU utilization and memory usage

## Troubleshooting

### Common Issues

1. **NCCL Communication Errors**
   - Check network connectivity
   - Verify firewall settings
   - Validate IP address configuration

2. **Insufficient GPU Memory**
   - Reduce batch_size
   - Use more GPUs
   - Enable KV cache optimization

3. **Model Loading Failures**
   - Check model path
   - Verify file integrity
   - Confirm permission settings

### Performance Optimization

1. **Enable MPS Service**
   ```bash
   nvidia-cuda-mps-control -d
   ```

2. **Enable Micro-batch Overlap**
   ```bash
   --enable_prefill_microbatch_overlap
   --enable_decode_microbatch_overlap
   ```

3. **Adjust CUDA Graph Parameters**
   ```bash
   --graph_max_batch_size 100
   ```

## Testing and Validation

### Basic Functionality Test

```bash
curl http://server_ip:server_port/generate \
     -H "Content-Type: application/json" \
     -d '{
           "inputs": "What is AI?",
           "parameters":{
             "max_new_tokens":17, 
             "frequency_penalty":1
           }
          }'
```

### Performance Benchmark Test

```bash
cd test
python benchmark_client.py \
--num_clients 100 \
--input_num 2000 \
--tokenizer_path /path/DeepSeek-R1/ \
--url http://127.0.0.1:8088/generate_stream
```

## Important Notes

1. Please modify the model path in scripts before use
2. Adjust parameters according to actual hardware configuration
3. Ensure network environment meets multi-node deployment requirements
4. Recommend thorough testing before production deployment
5. Regularly monitor service status and performance metrics 