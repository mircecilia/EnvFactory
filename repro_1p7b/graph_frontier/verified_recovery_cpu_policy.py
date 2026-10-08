"""Fresh, greedy ORIGINAL Dynamic-v1 inference on CPU only; official tool parser.

Both arms use this SAME engine. Not assumed bitwise equivalent to historic GPU
SGLang trajectories. No quantization, no optimization, no teacher LLM.
"""
import hashlib
import copy
import json
import os
import time
from pathlib import Path


class CPUPolicy:
    def __init__(self,model,threads=16):
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only launch contract missing'
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        torch.set_num_threads(threads);torch.set_num_interop_threads(1)
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=True)
        self.model=AutoModelForCausalLM.from_pretrained(model,local_files_only=True,trust_remote_code=True,
            torch_dtype=torch.float32,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cpu').eval()
        assert all(p.device.type=='cpu' for p in self.model.parameters())
        self.eos=self.model.generation_config.eos_token_id
        self.eos_ids=self.eos if isinstance(self.eos,list) else [self.eos]
        self.parsers={}
        self.cache={}
        self.prefix_ids=[]
        self.prefix_kv=None

    def generate(self,messages,schemas,seed,config):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.entrypoints.openai.protocol import Tool
        key=hashlib.sha256(json.dumps(schemas,sort_keys=True).encode()).hexdigest()
        if key not in self.parsers:self.parsers[key]=FunctionCallParser([Tool.model_validate(x) for x in schemas],'qwen25')
        ids=self.tokenizer.apply_chat_template(messages,tools=schemas,tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_tensors='pt')
        if ids.shape[-1]+config['max_tokens_per_turn']>32768:raise ValueError('frozen context limit exceeded')
        input_hash=hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest()
        cache_key=(input_hash,seed,config['max_tokens_per_turn'])
        if cache_key in self.cache:
            message,finish,audit=copy.deepcopy(self.cache[cache_key])
            audit.update(cached_identical_fresh_input=True,source_inference_seconds=audit['model_seconds'],model_seconds=0.0)
            return message,finish,audit
        class Progress:
            def __init__(self):self.first=True;self.tokens=0
            def put(self,value):
                if self.first:self.first=False;return
                self.tokens+=value.numel()
                if self.tokens%25==0:print(json.dumps({'cpu_generated_tokens':self.tokens,'seed':seed,'prompt_tokens':ids.shape[-1]}),flush=True)
            def end(self):pass
        common=0;full_ids=ids[0].tolist()
        # Cache is cropped against actual token IDs, never against gold state.
        # At least the last prompt token is re-fed to obtain next-token logits.
        while common<min(len(full_ids)-1,len(self.prefix_ids)) and full_ids[common]==self.prefix_ids[common]:common+=1
        kwargs={}
        if common and self.prefix_kv is not None:
            cache=copy.deepcopy(self.prefix_kv);cache.crop(common)
            assert cache.get_seq_length()==common
            kwargs['past_key_values']=cache
        else:common=0
        self.torch.manual_seed(seed);start=time.monotonic()
        with self.torch.inference_mode():
            outputs=self.model.generate(ids,attention_mask=self.torch.ones_like(ids),do_sample=False,
                max_new_tokens=config['max_tokens_per_turn'],eos_token_id=self.eos,
                pad_token_id=self.tokenizer.pad_token_id,use_cache=True,streamer=Progress(),
                return_dict_in_generate=True,**kwargs)
        generated=outputs.sequences[0,ids.shape[-1]:].tolist();ended=bool(generated and generated[-1] in self.eos_ids)
        self.prefix_kv=outputs.past_key_values
        self.prefix_ids=outputs.sequences[0,:self.prefix_kv.get_seq_length()].tolist()
        raw=self.tokenizer.decode(generated[:-1] if ended else generated,skip_special_tokens=False)
        content,calls=self.parsers[key].parse_non_stream(raw)
        message={'role':'assistant','content':content}
        if calls:
            message['tool_calls']=[{'id':f'cpu_{seed}_{i}','type':'function',
                'function':{'name':c.name,'arguments':json.loads(c.parameters)}} for i,c in enumerate(calls)]
        audit={'input_token_ids_sha256':input_hash,
            'generated_token_ids':generated,'raw_text':raw,'model_seconds':time.monotonic()-start,
            'prompt_tokens':ids.shape[-1],'completion_tokens':len(generated),'device':'cpu','dtype':'float32',
            'parser':'sglang.FunctionCallParser:qwen25','seed':seed,'cached_identical_fresh_input':False,'kv_prefix_tokens_reused':common}
        finish='stop' if ended or len(generated)<config['max_tokens_per_turn'] else 'length'
        self.cache[cache_key]=copy.deepcopy((message,finish,audit))
        return message,finish,audit
