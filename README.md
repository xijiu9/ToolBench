<div align= "center">
    <h1> 🛠️ToolBench🤖</h1>
</div>

<div align="center">

![Dialogues](https://img.shields.io/badge/Tool\_Num-3451-red?style=flat-square)
![Dialogues](https://img.shields.io/badge/API\_Num-16464-red?style=flat-square)
![Dialogues](https://img.shields.io/badge/Current\_Dataset\_Size-12K-red?style=flat-square)
![Dialogues](https://img.shields.io/badge/Total\_API\_Call-37K-red?style=flat-square)
![Dialogues](https://img.shields.io/badge/Average\_Reasoning\_Traces-4.1-red?style=flat-square)
![Dialogues](https://img.shields.io/badge/Tool\_LLaMA-Released-green?style=flat-square)

</div>

<p align="center">
  <a href="#model">Model</a> •
  <a href="#data">Data Release</a> •
  <a href="https://github.com/OpenBMB/BMTools">Toolkit</a> •
  <a href="https://arxiv.org/abs/2304.08354">Paper</a> •
  <a href="https://github.com/thunlp/ToolLearningPapers">Paper List</a> •
  <a href="#citation">Citation</a> •

</p>

</div>


🔨This project aims to construct **open-source, large-scale, high-quality** instruction tuning SFT data to facilitate the construction of powerful LLMs with general **tool-use** capability. We aim to empower open-source LLMs to master thousands of diverse real-world APIs. We achieve this by collecting a high-quality instruction-tuning dataset. It is constructed automatically using the latest ChatGPT (gpt-3.5-turbo-16k), which is upgraded with enhanced [function call](https://openai.com/blog/function-calling-and-other-api-updates) capabilities. We provide the dataset, the corresponding training and evaluation scripts, and a capable model ToolLLaMA fine-tuned on ToolBench.

✨More details and our paper about ToolBench and ToolLLaMA are coming soon!

<div align="center">
<img src="https://cdn.discordapp.com/attachments/941582479117127680/1111543600879259749/20230526075532.png" width="400px">
</div>

✨✨Features:
 - API Collection: we gather **16464** representational state transfer (REST) APIs from [RapidAPI](https://rapidapi.com/hub), a platform that hosts massive real-world APIs provided by developers.
 - Instruction Generation: we curate instructions that involve both **single-tool** and **multi-tool** scenarios.
 - Answer Annotation: we develop a novel **depth-first search based decision tree** (DFSDT) to bolster the planning and reasoning ability of LLMs, which significantly improves the annotation efficiency and successfully annotates those complex instructions that cannot be answered with CoT or ReACT. We provide responses that not only include the final answer but also incorporate the model's **reasoning process, tool execution, and tool execution results**. 
 - API Retriver: we incorporate API retrieval to equip ToolLLaMA with open-domain tool-using abilities.
 - All the data is automatically generated by OpenAI API and filtered by us, the whole data creation process is easy to scale up.


<br>
<div align="center">
<img src="assets/overview.png" width="800px">
</div>
<br>

*Please note that current released data is still not the final version. We are conducting extensive post-processing to improve the data quality and increase the coverage of real-world tools.*

*[Old version](https://github.com/OpenBMB/ToolBench/tree/legacy)*
<!-- 💁‍♂️💁💁‍♀️**We need your help!** Curating large-scale real-world APIs and their corresponding tool-use SFT data is not easy, we sincerely invite you to join us in building and refining ToolBench. We will list all participants as co-authors in the final paper. Please contact and join [us](mailto:yujiaqin16@gmail.com) if you're interested. -->

## 🗒️Data

👐ToolBench is intended solely for research and educational purposes and should not be construed as reflecting the opinions or views of the creators, owners, or contributors of this dataset. It is distributed under [CC BY NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/). Below is the statistics of the data :

| Tool Nums | API Nums | Instance Nums | Real API Call | Reasoning Traces |
|-----------|----------|---------------|---------------|------------------|
| 3451      | 16464    | 12657         | 37204         | 4.1              |


ToolBench contains both single-tool and multi-tool scenarios. The multi-tool scenarios can be further categorized into intra-category multi-tool and intra-collection multi-tool. We utilize DFSDT method for all scenarios to our data creation. Here is an example of the data creation and application process using BMTools:

<div align="center">

<img src="assets/meta0423.gif" width="700px">

</div>

### Data Release

 Please download our dataset using the following link: [Data](https://drive.google.com/drive/folders/1yBUQ732mPu-KclJnuQELEhtKakdXFc3J).
- `G1`,` G2`, `G3`data refers to single-tool, intra-category multi-tool and intra-collection multi-tool data respectively.
- We split the G1, G2 and G3 data into train, eval and test parts respectively and combine the train data for training in our main experiments. `toolllama_G123_dfs_train.json` refers to the combined train data.
- The tool environment related data is in `toolenv` directory.
- We sample 100 instances from every test set. The `test_query_ids` directory contains query ids of the test instances in each test set.
- The data used for tool retrieval is also included in the `retrieval` directory.


## 🤖Model

We release the 7b lora version of [ToolLLaMA](https://huggingface.co/pooruss/ToolLLaMA-7b-lora) which is trained on the released dataset. The models are trained in a multi-task fashion.

## 🚀Fine-tuning
### Install
Clone this repository and navigate to the ToolBench folder.
```bash
git clone git@github.com:OpenBMB/ToolBench.git
cd ToolBench
```
Install Package (python>=3.9)
```bash
pip install -r requirements.txt
```

### Data Preparation

Download the [data](https://drive.google.com/drive/folders/1yBUQ732mPu-KclJnuQELEhtKakdXFc3J) and unzip it under ToolBench.


### Training Retriever
- Data preprocessing:
```bash
export PYTHONPATH=./
python data/preprocess_retriever_data.py \
    --query_file data/instruction/G1_query.json \
    --index_file data/test_query_ids/G1_instruction_test_query_ids.json \
    --dataset_name G1 \
    --output_dir data/retrieval/G1
```
- Then run the following command to train the tool retriever:
```bash
export PYTHONPATH=./
python toolbench/retrieval/train.py \
    --data_path data/retrieval/G1/ \
    --model_name bert-base-uncased \
    --output_path retrieval_model \
    --num_epochs 5 \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --warmup_steps 500 \
    --max_seq_length 256
```

### Training ToolLLaMA
Our training code is based on [FastChat](https://github.com/lm-sys/FastChat). You can use the following command to train ToolLLaMA-7b with 2 x A100 (80GB), with the preprocessed data in our [data link](https://drive.google.com/drive/folders/1yBUQ732mPu-KclJnuQELEhtKakdXFc3J). We split the G1, G2 and G3 data into train, eval, test parts respectively and combine the train data for training. 

You can also preprocess the data and split in your own way with this command:
```bash
export PYTHONPATH=./
python preprocess/preprocess_toolllama_data.py \
    --tool_data_dir data/answer/G1_answer \
    --method DFS_woFilter_w2 \
    --output_file data/answer/toolllama_G1_dfs.json
```

To train ToolLLaMA, use the following command:
```bash
export PYTHONPATH=./
torchrun --nproc_per_node=2 --master_port=20001 toolbench/train/train_long_seq.py \
    --model_name_or_path huggyllama/llama-7b  \
    --data_path  data/toolllama_G123_dfs_train.json \
    --eval_data_path  data/toolllama_G123_dfs_eval.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir toolllama \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --prediction_loss_only \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none
```
And train with lora:
```bash
export PYTHONPATH=./
deepspeed --master_port=20001 toolbench/train/train_long_seq_lora.py \
    --model_name_or_path huggyllama/llama-7b  \
    --data_path  data/toolllama_G123_dfs_train.json \
    --eval_data_path  data/toolllama_G123_dfs_eval.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir toolllama_lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "epoch" \
    --prediction_loss_only \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \    
    --deepspeed ds_configs/stage2.json \
    --report_to none

```


## Inference

First prepare your rapidapi key:
```bash
export RAPIDAPIKEY="your_rapidapi_key"
```

Then run the following commands:
```bash
export PYTHONPATH=./
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model toolllama \
    --model_path toolllama \
    --max_observation_length 1024 \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo.json \
    --output_answer_file data/answer/toolllama_dfs \
    --rapidapi_key $RAPIDAPIKEY
```

For **lora** version:
```bash
export PYTHONPATH=./
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model toolllama \
    --model_path huggyllama/llama-7b \
    --lora \
    --lora_path toolllama_lora \
    --max_observation_length 1024 \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo.json \
    --output_answer_file data/answer/toolllama_lora_dfs \
    --rapidapi_key $RAPIDAPIKEY
```

For lora version under **open-domain** setting, run:
```bash
export PYTHONPATH=./
python toolbench/inference/qa_pipeline_open_domain.py \
    --tool_root_dir data/toolenv/tools/ \
    --corpus_tsv_path data/retrieval/G1/corpus.tsv \
    --retrieval_model_path retrival_model \
    --retrieved_api_nums 5 \
    --backbone_model toolllama \
    --model_path huggyllama/llama-7b \
    --lora \
    --lora_path toolllama_lora \
    --max_observation_length 1024 \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo_open_domain.json \
    --output_answer_file data/answer/toolllama_lora_dfs_open_domain \
    --rapidapi_key $RAPIDAPIKEY
```


## ToolEval

By fine-tuning LLaMA on ToolBench, we obtain **ToolLLaMA**. Considering that human evaluation can be time-consuming, we follow [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) to develop an efficient machine evaluator **ToolEval**, which incorporates two evaluation metrics:
 - **Pass Rate**: Calculates the proportion of successfully completing an instruction within limited OpenAI API calls. 
 - **Preference**: Measured by comparing two answers (action sequences) for a given instruction. We pre-define a set of criteria for a better answer, which are organized as prompts for ChatGPT. We provide the test instruction and two candidate answers to the evaluator and obtain its preference. We evaluate each answer pair multiple times to improve the reliability of our system. Then we calculate the **Win Rate** (percentage of being preferred by the evaluator) and **Standard Error** (the standard error of the Win Rate). More details can be found in our paper.

To validate the effectiveness of the metric **Preference**, we sample among three different methods (ChatGPT+ReACT, GPT4+ReACT, and ChatGPT+DFSDT) to obtain answer pairs for *600* test instructions. Then we engage humans to annotate human preference for them (*4* annotations for each answer pair, *2400* annotations in total).
Our automatic evaluator, developed using \turbo, demonstrates a significant correlation of **75.8%** with human annotators.
We also obtain the agreement among different human annotators **83.54%**, and the agreement between humans and our evaluator **80.21%**.

More details about ToolEval can be found in our paper.

### Evaluation with ToolEval
To evaluate a model on G1-Inst. test set, for example, run the following commands.
- Pass rate:
```bash
python toolbench/tooleval/pass_rate.py --answer_dir data/answer/toolllama_dfs/G1_instruction
```
- Win rate (Reference model: ChatGPT-ReACT):
```bash
export OPENAI_KEY=""
export REF_MODEL_DATA="data/answer/chatgpt_cot/G1_instruction"
export REF_MODEL_METHOD="CoT"
export TEST_MODEL_DATA="data/answer/toolllama_dfs/G1_instruction"
export TEST_MODEL_METHOD="DFS"
python ./toolbench/tooleval/convert_to_answer_format.py \
    --method CoT \
    --answer_dir $REF_MODEL_DATA \
    --output ${REF_MODEL_DATA}_converted

python ./toolbench/tooleval/convert_to_answer_format.py \
    --method DFS \
    --answer_dir $TEST_MODEL_DATA \
    --output ${TEST_MODEL_DATA}_converted

python ./toolbench/tooleval/automatic_eval_sample.py \
    --output ${REF_MODEL_DATA}_converted \
    --ref_output ${TEST_MODEL_DATA}_converted \
    --method $REF_MODEL_METHOD \
    --use_existed_output
```

### Model Experiment

In our main experiments, ToolLLaMA demonstrates a compelling capability to handle both single-tool and complex multi-tool instructions.
Below are the main results compared with ChatGPT and Text-Davinci-003.

**Pass Rate:**
| model                  | I1-Inst. | I1-Tool. | I1-Cat. | I2-Inst. | I2-Cat. | I3-Inst. | Average |
|------------------------|----------|----------|---------|----------|---------|----------|---------|
| ChatGPT-ReACT          | 66       | 56       | 62      | 22       | 28      | 30       | 44.0    |
| ChatGPT-DFSDT          | **89**       | **78**       | **84**      | **58**       | **51**      | **57**       | **69.6**    |
| Text-Davinci-003-DFSDT | 61       | 53       | 58      | 38       | 38      | 39       | 47.8    |
| ToolLLaMA              | 75       | 68       | 80      | 56       | 47      | 40       | 61.0    |

**Win Rate:** (Reference model: ChatGPT-DFSDT)
| model                  | I1-Inst. | I1-Tool. | I1-Cat. | I2-Inst. | I2-Cat. | I3-Inst. | Average |
|------------------------|----------|----------|---------|----------|---------|----------|---------|
| Text-Davinci-003-DFSDT | 38       | 34       | 43      | 25       | 20      | 28       | 31.3    |
| ToolLLaMA              | **50**       | 45       | 45      | **59**       | 48      | 46       | 48.8    |


## TODO
- [ ] Release the rest part of the data for other tools in BMTools.
- [ ] ToolLLaMA will reach GPT-4's tool-use capability.
- [ ] We will train a ToolLLaMa-2

## Citation
Feel free to cite us if you like ToolBench.

```bibtex
@misc{qin2023tool,
      title={Tool Learning with Foundation Models}, 
      author={Yujia Qin and Shengding Hu and Yankai Lin and Weize Chen and Ning Ding and Ganqu Cui and Zheni Zeng and Yufei Huang and Chaojun Xiao and Chi Han and Yi Ren Fung and Yusheng Su and Huadong Wang and Cheng Qian and Runchu Tian and Kunlun Zhu and Shihao Liang and Xingyu Shen and Bokai Xu and Zhen Zhang and Yining Ye and Bowen Li and Ziwei Tang and Jing Yi and Yuzhang Zhu and Zhenning Dai and Lan Yan and Xin Cong and Yaxi Lu and Weilin Zhao and Yuxiang Huang and Junxi Yan and Xu Han and Xian Sun and Dahai Li and Jason Phang and Cheng Yang and Tongshuang Wu and Heng Ji and Zhiyuan Liu and Maosong Sun},
      year={2023},
      eprint={2304.08354},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
