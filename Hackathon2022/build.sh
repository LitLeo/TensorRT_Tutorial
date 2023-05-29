cd LayerNormPlugin && make && cd ..
python encoder2trt.py && python decoder2trt.py && cp LayerNormPlugin/*.so .
