# prompt cache 测试：

- benchmark_prompt_cache.py： 单次测试脚本。

    例子：
    ```shell
    python benchmark_prompt_cache.py --address http://localhost:8090 --model_name llama --num_workers 1 --first_input_len 512 --subsequent_input_len 32 --output_len 32 --num_turns 5 --num_users 1
    ```

    使用方法详细说明： 
    ```shell
    python benchmark_prompt_cache.py -h
    ```

- test_settings.py： 批量测试脚本，可测试多个配置并汇总为md
