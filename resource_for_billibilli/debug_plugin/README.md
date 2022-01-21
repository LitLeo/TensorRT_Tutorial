1. debug_plugin demo for debug, to print output of each layer.
2. support TensorRT 6&7 

# build
```
mkdir build && cd build
export TRT_RELEASE=/your/trt/path
cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out
```
