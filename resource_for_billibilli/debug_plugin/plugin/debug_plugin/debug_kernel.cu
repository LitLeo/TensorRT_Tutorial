#include "debug_kernel.h"
#include <numeric>

using namespace nvinfer1;
using namespace std;

namespace debug_plugin {

template<class T>
void p_row(const T *row_ptr, int row, int width, int h, bool fold, int t)
{
  // save old format
  ios::fmtflags f(cout.flags()); 
  cout.flags(ios::left);

  // print blank
  if (row != 0) printf("%*s", t - 1, "");
  if (width > 6 && fold) {
    // char/uint8_t or float32
    if (sizeof(T) == 1)
      cout << "[row_" << row << "]: [" << (intptr_t)row_ptr[0] << ", " << (intptr_t)row_ptr[1] <<
           ", " << (intptr_t)row_ptr[2] << ", " << "..., " << (intptr_t)row_ptr[width-3] << ", " <<
           (intptr_t)row_ptr[width-2] << ", " << (intptr_t)row_ptr[width-1] << "]";
    else
      cout << setprecision(8) << "[row_" << row << "]: [" << row_ptr[0] << ", " << row_ptr[1] <<
           ", " << row_ptr[2] << ", " << "..., " << row_ptr[width-3] << ", " <<
           row_ptr[width-2] << ", " << row_ptr[width-1] << "]";
  } else {
    cout << "[row_" << row << "]: [";
    for (size_t i = 0; i < width; i++) {
      // char/uint8_t or float32
      if (sizeof(T) == 1) cout << (intptr_t)row_ptr[i];
      else cout << setprecision(8) << row_ptr[i];
      if (i != width - 1) cout << ", ";
    }
    cout << "]";
  }
  if (row != h - 1) cout << "," << endl;
  // restore old format
  cout.flags(f);
}

// for debug
template<class T>
void p(const T *data, int height, int width, bool fold_tensor, int dims_t) {
  cout << "[";
  if (height > 6) {
    p_row(data, 0, width, height, fold_tensor, dims_t);
    p_row(data + width * 1, 1, width, height, fold_tensor, dims_t);
    p_row(data + width * 2, 2, width, height, fold_tensor, dims_t);
    printf("%*s", dims_t - 1, "");
    printf("... ...,\n");
    p_row(data + width * (height - 3), height - 3, width, height, fold_tensor, dims_t);
    p_row(data + width * (height - 2), height - 2, width, height, fold_tensor, dims_t);
    p_row(data + width * (height - 1), height - 1, width, height, fold_tensor, dims_t);
  } else {
    for (size_t i = 0; i < height; i++)
      p_row(data + width * (i), i, width, height, fold_tensor, dims_t);
  }
  cout << "]";
}

/*
template<class T>
void p(const T *data, int idx, vector<int>& dims) {
  if (dims.size() - idx == 2) {
    p(data, dims[idx], dims[idx + 1]);
  } else {
    cout << "[" << endl;
    int offset = 1;
    for (size_t i = idx + 1; i < dims.size(); i++) 
      offset *= dims[i];
      p(data + offset, idx + 1, dims);
    cout << "]" << endl;
  }
}
*/

template<class T>
void p(const T *data, vector<int>& dims) {
  // print Tensor shape
  int dims_size = dims.size();
  cout << "Dims: [";
  for (int i = 0; i < dims_size; i++) {
    cout << dims[i];
    if (i != dims_size - 1)
      cout << ", ";
  }
  cout << "]" << endl;
  
  // 对于一维和二维Tensor，行元素数量超过15才会压缩
  bool fold_tensor = true;
  if (dims_size == 1 && dims[0] < 15) fold_tensor = false;
  else if (dims_size == 2) {
    if (dims[0] <= 6 && dims[1] < 15) fold_tensor = false;
  }

  if (dims_size == 1) {
    p_row(data, 0, dims[0], 1, fold_tensor, 1);
  } else if (dims_size == 2) {
    p(data, dims[0], dims[1], fold_tensor, 2);
    printf("\n");
  } else if (dims_size > 2 && dims_size <= 6) {
    // fix max dim_size to 6 and padding tensor to 6-dims
    int len = 6;
    for (int m = 0; m < len - dims_size; m++)
      dims.insert(dims.begin(), 1);
    int offset = 0;
    int block = dims[len - 1] * dims[len - 2];
    int block_3d = dims[len - 1] * dims[len - 2] * dims[len - 3];
    int block_4d = dims[len - 1] * dims[len - 2] * dims[len - 3] * dims[len - 4];
    int block_5d = dims[len - 1] * dims[len - 2] * dims[len - 3] * dims[len - 4] * dims[len - 5];

    // =============== 6-D ===============
    if (dims_size >= 6) cout << "[";
    for (int l = 0; l < dims[len - 6]; l++) {
      // decide whether to fold 6-D matrix
      bool show6 = false, fold6 = true;
      if (dims[len - 6] <= 6) {show6 = true; fold6 = false;}
      else if (l < 3 || l >= dims[len - 6] - 3) {show6 = true; fold6 = true;}
      // print 6-D blank
      if (show6 && l > 0) printf("%*s", dims_size - 5, "");
      // print 6-D fold line
      if (fold6 && l == 3) printf("%*s...... ......\n\n\n\n\n", dims_size - 5, "");
      // =============== 5-D ===============
      if (show6 && dims_size >= 5) cout << "[";
      for (int k = 0; k < dims[len - 5]; k++) {
        // decide whether to fold 5-D matrix
        bool show5 = false, fold5 = true;
        if (dims[len - 5] <= 6) {show5 = true; fold5 = false;}
        else if (k < 3 || k >= dims[len - 5] - 3) {show5 = true; fold5 = true;}
        show5 = show5 && show6;
        // print 5-D blank
        if (show5 && k > 0) printf("%*s", dims_size - 4, "");
        // print 5-D fold line
        if (fold5 && k == 3 && l != 3) printf("%*s..... .....\n\n\n\n", dims_size - 4, "");
        // =============== 4-D ===============
        if (show5 && dims_size >= 4) cout << "[";
        for (int j = 0; j < dims[len - 4]; j++) {
          // decide whether to fold 4-D matrix
          bool show4 = false, fold4 = true;
          if (dims[len - 4] <= 6) {show4 = true; fold4 = false;}
          else if (j < 3 || j >= dims[len - 4] - 3) {show4 = true; fold4 = true;}
          show4 = show4 && show5 && show6;
          // print 4-D blank
          if (show4 && j > 0) printf("%*s", dims_size - 3, "");
          // print 4-D fold line
          if (fold4 && j == 3 && k != 3 && l != 3) printf("%*s.... ....\n\n\n", dims_size - 3, "");
          // =============== 3-D ===============
          if (show4 && dims_size >= 3) cout << "[";
          for (int i = 0; i < dims[len - 3]; i++) {
            // decide whether to fold 3-D matrix
            bool show3 = false, fold3 = true;
            if (dims[len - 3] <= 6) {show3 = true; fold3 = false;}
            else if (i < 3 || i >= dims[len - 3] - 3) {show3 = true; fold3 = true;}
            show3 = show3 && show4 && show5 && show6;
            // print 3-D blank
            if (show3 && i > 0) printf("%*s", dims_size - 2, "");
            if (fold3 && i == 3 && j != 3 && k != 3 && l != 3) printf("%*s... ...\n\n", dims_size - 2, "");
            // compute offset
            offset = l * block_5d + k * block_4d + j * block_3d + i * block;
            // =============== 2-D ===============
            if (show3)
              p(data + offset, dims[len - 2], dims[len - 1], fold_tensor, dims_size);
            // =============== 2-D End ===============
            if ((fold3 && show3 && (i < 3 || i == dims[len - 3] - 3 || i == dims[len - 3] - 2)) || 
                (!fold3 && show3 && i != dims[len - 3] - 1))
              printf("\n\n");
          }
          // =============== 3-D End ===============
          if ((fold4 && show4 && (j < 3 || j == dims[len - 4] - 3 || j == dims[len - 4] - 2)) || 
              (!fold4 && show4 && j != dims[len - 4] - 1))
            printf("]\n\n\n");
        }
        // =============== 4-D End ===============
        if ((fold5 && show5 && (k < 3 || k == dims[len - 5] - 3 || k == dims[len - 5] - 2)) || 
            (!fold5 && show5 && k != dims[len - 5] - 1))
          printf("]]\n\n\n\n");
      }
      // =============== 5-D End ===============
      if ((fold6&& show6 && (l < 3 || l == dims[len - 6] - 3 || l == dims[len - 6] - 2)) || 
          (!fold6 && show6 && l != dims[len - 6] - 1))
        printf("]]]\n\n\n\n\n");
    }
    // =============== 6-D End ===============
    for (int m = 0; m < dims_size - 2; m++)
      cout << "]";
    cout << endl;
  } else {
    cout << "Tensor dims num must <= 6!" << endl;
  }
}

void p(const float *data, vector<int>& dims) {
  p<float>(data, dims);
}


void p_sum(const float *data, std::vector<int>& dims, string message) {
  int batch_size = dims[0];
  int sum_of_elems = 1;
  for (size_t i = 1; i < dims.size(); i++) 
    sum_of_elems *= dims[i];

  // fix size=1
  if (dims.size() == 1) {
    batch_size = 1;
    sum_of_elems = dims[0];
  }

  std::cout << message << ", batch_size = " << batch_size << ", sum= :";
  for (int i = 0; i < batch_size; i++) {
    float sum = 0.0f;
    const float *p_ptr = data + i * sum_of_elems;
    for (int j = 0; j < sum_of_elems; j++) 
      sum += p_ptr[j];
    cout << sum << ", ";
  }
  std::cout << std::endl;

}


} // debug_plugin

