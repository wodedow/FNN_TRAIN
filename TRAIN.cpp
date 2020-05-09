#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "Struct.h"

using namespace std;

const int batch = 5;
const int batch_sizes = 10;	    //nums of dataes
const int img_size = 28 * 28;	//sizes of data
const int classes = 10;	    	//nums of classes
const int times = 30;	    	//nums of iterations
const int update = 16;

const char add_imgs[] = "D:\\MnistHandWriting\\train-images.idx3-ubyte";
const char add_labs[] = "D:\\MnistHandWriting\\train-labels.idx1-ubyte";

/*************************************************************************************/
void createDatas(List2& weight_arrays, List& bias, int L, int* m) {
    //初始化创建数据：List2& weight_arrays, List& bias
    //随机创建权重 W
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < m[i + 1]; j++) {
            for (int k = 0; k < m[i]; k++) {
                double s = (float)rand() / RAND_MAX - 0.5;
                weight_arrays.List2_elem[i].list2_elem[j][k] = 2.0 * s;
                //cout << weight_arrays.List2_elem[i].list2_elem[j][k] << endl;
                //printf("%f:::", weight_arrays.List2_elem[i].list2_elem[j][k]);
            }
        }
    }
    //随机创建偏置 b
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < m[i + 1]; j++) {
            double t = (float)rand() / RAND_MAX - 0.5;
            bias.List_elem[i].list_elem[j] = 2.0 * t;
            //cout << "bias" << i<< ":" << bias.List_elem[i].list_elem[j] << endl;
        }
    }
    printf("Function createDatas is run!!!\n");
    //cout << "Function createDatas is run" << endl;
}
int swap(int x)
{
    return (((int)(x) & 0xff000000) >> 24) | \
        (((int)(x) & 0x00ff0000) >> 8) | \
        (((int)(x) & 0x0000ff00) << 8) | \
        (((int)(x) & 0x000000ff) << 24);
}
long* readDataFromMnist_train(double** images_train, double** labels_train, int batch_size = batch_sizes) {
    FILE* imgs;
    FILE* labs;
    errno_t err_img = 0; errno_t err_lab = 0;
    err_img = fopen_s(&imgs, add_imgs, "rb");
    err_lab = fopen_s(&labs, add_labs, "rb");
    if (err_img != 0) {
        printf("Can't Open This File: Images!!!\n");
    }
    if (err_img != 0) {
        printf("Can't Open This File: Labels!!!\n");
    }

    int magic;				//文件中的魔术数(magic number)  
    int num_items;			//mnist图像集文件中的图像数目  
    int num_label;			//mnist标签集文件中的标签数目  
    int rows;				//图像的行数  
    int cols;				//图像的列数  

    fread(&magic, sizeof(int), 1, imgs);
    if (swap(magic) != 2051) {
        printf("This isn't the Mnist Images File!!!\n");
    }
    fread(&magic, sizeof(int), 1, labs);
    if (swap(magic) != 2049) {
        printf("This isn't the Mnist Labels File!!!\n");
    }

    fread(&num_items, sizeof(int), 1, imgs);
    fread(&num_label, sizeof(int), 1, labs);
    if (swap(num_items) != swap(num_label)) {
        printf("The Image File and Label File are not a Pair!!!\n");
    }

    fread(&rows, sizeof(int), 1, imgs);
    fread(&cols, sizeof(int), 1, imgs);
    rows = swap(rows); cols = swap(cols);

    int sizes = rows * cols;
    if (sizes != img_size) {
        printf("The Size of Picture is False!!!\n");
    }

    char* pixels_img = new char[sizes];
    char label;
    //int c;
    double mn;
    for (int i = 0; i < batch_sizes; i++) {
        fread(pixels_img, sizeof(char), sizes, imgs);
        fread(&label, sizeof(char), 1, labs);
        //图像数据
        for (int j = 0; j < sizes; j++) {
            mn = pixels_img[j];
            //cout<< m << "--";
            if (mn == 0) {
                images_train[i][j] = 0;
                //printf("%d***", (int)images_train[i][j]);
            }
            else {
                images_train[i][j] = 1;
                //printf("%d***", (int)images_train[i][j]);
            }
            //printf("%f***", mn);
        }
        //标签数据
        //printf("++++++++++++++++++++++++++\n");
        int classes_k = (int)label;
        //printf("%d\n", classes_k);
        for (int k = 0; k < classes; k++) {
            if (k == classes_k)
                labels_train[i][k] = 1;
            else
                labels_train[i][k] = 0;
        }
        //printf("\n");
    }
    delete[] pixels_img;
    long* offset = new long[2];
    offset[0] = ftell(imgs);
    //printf("The position of imgs_pre is %ld\n", offset[0]);
    offset[1] = ftell(labs);
    //printf("The position of labs_pre is %ld\n", offset[1]);
    fclose(imgs);
    fclose(labs);
    return offset;
}
long* readDataFromMnist_train_con(double** images_train, double** labels_train, long* offset, int batch_size = batch_sizes) {
    FILE* imgs;
    FILE* labs;
    errno_t err_img = 0; errno_t err_lab = 0;
    err_img = fopen_s(&imgs, add_imgs, "rb");
    err_lab = fopen_s(&labs, add_labs, "rb");

    int sizes = img_size;
    char* pixels_img = new char[sizes];
    char label;
    //int c;
    fseek(imgs, offset[0], SEEK_CUR);
    printf("The position of imgs_con is %ld\n", ftell(imgs));
    fseek(labs, offset[1], SEEK_CUR);
    printf("The position of labs_con is %ld\n", ftell(labs));
    for (int i = 0; i < batch_sizes; i++) {
        fread(pixels_img, sizeof(char), sizes, imgs);
        fread(&label, sizeof(char), 1, labs);
        //图像数据
        for (int j = 0; j < sizes; j++) {
            double mn = pixels_img[j];
            //cout<< m << "--";
            if (mn == 0) {
                images_train[i][j] = 0;
                //printf("%d***", (int)images_train[i][j]);
            }
            else {
                images_train[i][j] = 1;
                //printf("%d***", (int)images_train[i][j]);
            }
            //printf("%f***", mn);
        }
        //标签数据
        //printf("++++++++++++++++++++++++++\n");
        int classes_k = (int)label;
        //printf("%d\n", classes_k);
        for (int k = 0; k < classes; k++) {
            if (k == classes_k)
                labels_train[i][k] = 1;
            else
                labels_train[i][k] = 0;
        }
        //printf("\n");
    }
    delete[] pixels_img;
    long* offset_cur = new long[2];
    offset_cur[0] = ftell(imgs);
    offset_cur[1] = ftell(labs);
    fclose(imgs);
    fclose(labs);
    return offset_cur;
}
long* readDataFromMnist_random(double** images_train, double** labels_train, int batch_size = batch_sizes) {
    FILE* fin_imgs;
    FILE* fin_labs;
    errno_t err_img = 0; errno_t err_lab = 0;
    err_img = fopen_s(&fin_imgs, "imgs_random.txt", "r");
    err_lab = fopen_s(&fin_labs, "labs_random.txt", "r");
    if (err_img != 0) {
        printf("Can't Open This File: Images!!!\n");
    }
    if (err_lab != 0) {
        printf("Can't Open This File: Labels!!!\n");
    }

    int classes_k;
    for (int k = 0; k < batch_size; k++) {
        for (int i = 0; i < img_size; i++) {
            fscanf_s(fin_imgs, "%lf", &images_train[k][i]);
        }
        fscanf_s(fin_labs, "%d", &classes_k);
        for (int i = 0; i < classes; i++) {
            if (i == classes_k)
                labels_train[k][i] = 1;
            else
                labels_train[k][i] = 0;
        }
    }

    long* offset = new long[2];
    offset[0] = ftell(fin_imgs);
    //printf("The position of imgs_pre is %ld\n", offset[0]);
    offset[1] = ftell(fin_labs);
    //printf("The position of labs_pre is %ld\n", offset[1]);
    fclose(fin_imgs);
    fclose(fin_labs);
    return offset;
}
long* readDataFromMnist_random_con(double** images_train, double** labels_train, long* offset, int batch_size = batch_sizes) {
    FILE* fin_imgs;
    FILE* fin_labs;
    errno_t err_img = 0; errno_t err_lab = 0;
    err_img = fopen_s(&fin_imgs, "imgs_random.txt", "r");
    err_lab = fopen_s(&fin_labs, "labs_random.txt", "r");

    fseek(fin_imgs, offset[0], SEEK_CUR);
    printf("The position of imgs_con is %ld\n", ftell(fin_imgs));
    fseek(fin_labs, offset[1], SEEK_CUR);
    printf("The position of labs_con is %ld\n", ftell(fin_labs));

    int classes_k;
    for (int k = 0; k < batch_size; k++) {
        for (int i = 0; i < img_size; i++) {
            fscanf_s(fin_imgs, "%lf", &images_train[k][i]);
        }
        fscanf_s(fin_labs, "%d", &classes_k);
        for (int i = 0; i < classes; i++) {
            if (i == classes_k)
                labels_train[k][i] = 1;
            else
                labels_train[k][i] = 0;
        }
    }

    long* offset_cur = new long[2];
    offset_cur[0] = ftell(fin_imgs);
    offset_cur[1] = ftell(fin_labs);
    fclose(fin_imgs);
    fclose(fin_labs);
    return offset_cur;
}
double** matrix_add(double** a, double* b, int j, int* nodes, int batch_size = batch_sizes) {
    //两矩阵相加
    double** c = new double* [batch_size];
    for (int ii = 0; ii < batch_size; ii++) {
        c[ii] = new double[nodes[j]];
    }
    for (int mm = 0; mm < batch_size; mm++) {
        for (int n = 0; n < nodes[j]; n++) {
            c[mm][n] = a[mm][n] + b[n];
        }
    }
    return c;
}
double** matrix_rot(double** a, double** b, int j, int* nodes, int batch_size = batch_sizes) {
    //两矩阵相乘
    double** c = new double* [batch_size];
    for (int ii = 0; ii < batch_size; ii++) {
        c[ii] = new double[nodes[j]];
    }
    for (int mm = 0; mm < batch_size; mm++) {
        for (int n = 0; n < nodes[j]; n++) {
            c[mm][n] = 0;
            for (int jj = 0; jj < nodes[j - 1]; jj++) {
                c[mm][n] += a[n][jj] * b[mm][jj];
            }
        }
    }
    return c;
}
double** sigmod_l(double** a, int j, int* nodes, int batch_size = batch_sizes) {
    //Sigmod函数：Logistic $\sigma(x)=\frac{1}{1+exp(-x)}$
    //RELU函数：$f(x)=\begin{cases}x \quad x \geq 0 \\ 0 \quad x<0 \end{cases}$
    double** c = new double* [batch_size];
    for (int ii = 0; ii < batch_size; ii++) {
        c[ii] = new double[nodes[j]];
    }
    for (int mm = 0; mm < batch_size; mm++) {
        for (int n = 0; n < nodes[j]; n++) {
            //c[n][mm] = (exp(a[n][mm])- exp(-a[n][mm])) / (exp(a[n][mm]) + exp(-a[n][mm]));
            c[mm][n] = 1.0 / (1 + exp(-a[mm][n]));
            /*if (a[mm][n] < 0)
                c[mm][n] = 0;
            else
                c[mm][n] = a[mm][n];*/
        }
    }
    return c;
}
double diff(double a) {
    //导函数 $ \sigma'(x)=\frac{exp(x)}{(1+exp(x))^2} $
    //导函数 $ f'(x)=\begin{cases}0 \quad x \leq 0\\1 \quad x>0 $
    double c;
    c = exp(a) / pow(1 + exp(a), 2);
    //if (a <= 0) {
    //    c = 0;
    //}
    //else {
    //    c = 1;
    //}
    return c;
}
double forward(FNN& fnn, int batch_size = batch_sizes) {
    //正向传递更新结点输入输出：List2& layers_in, List2& layers，并返回Loss
    int L = fnn.length - 1;
    for (int j = 1; j < L + 1; j++) {
        double** p;
        p = matrix_rot(fnn.weight_arrays.List2_elem[j - 1].list2_elem, fnn.layers.List2_elem[j - 1].list2_elem, j, fnn.nodes);
        double** q;
        q = matrix_add(p, fnn.bias.List_elem[j - 1].list_elem, j, fnn.nodes);
        fnn.layers_in.List2_elem[j - 1].list2_elem = q;
        fnn.layers.List2_elem[j].list2_elem = sigmod_l(q, j, fnn.nodes);
    }

    //for (int i = 0; i < 10; i++) {
    //    printf("%f  ", fnn.layers.List2_elem[L].list2_elem[0][i]);
    //}
    //printf("\n");
    //for (int i = 0; i < 10; i++) {
    //    printf("%f  ", fnn.layers.List2_elem[L].list2_elem[1][i]);
    //}
    //printf("\n");
    //for (int i = 0; i < 10; i++) {
    //    printf("%f  ", fnn.layers.List2_elem[L].list2_elem[2][i]);
    //}
    //printf("\n");
    //for (int i = 0; i < 10; i++) {
    //    printf("%f  ", fnn.layers.List2_elem[L].list2_elem[3][i]);
    //}
    //printf("\n");
    //for (int i = 0; i < 10; i++) {
    //    printf("%f  ", fnn.layers.List2_elem[L].list2_elem[4][i]);
    //}
    //printf("\n");

    double loss = 0;
    for (int j = 0; j < batch_size; j++) {
        for (int i = 0; i < fnn.nodes[L]; i++) {
            loss += 0.5 * pow(fnn.layers.List2_elem[L].list2_elem[j][i] - fnn.class_arrays[j][i], 2);
        }
    }

    fnn.first = false;
    printf("Function forward is run\n");
    printf("loss: %f\n", loss);
    return loss;
}

void backward(BPA& bpa, FNN& fnn) {
    //反向传播并存储相应导函数矩阵：diff_weight_arrays, diff_bias
    int L = bpa.length;
    int batch_size = bpa.batch_size;
    int* m = fnn.nodes;
    for (int k = L - 1; k >= 0; k--) {
        if (k == L - 1) {
            for (int j = 0; j < m[L]; j++) {
                for (int i = 0; i < m[L - 1]; i++) {
                    double a = 0;
                    for (int pm = 0; pm < batch_size; pm++) {
                        bpa.diff_layersin.List2_elem[L - 1].list2_elem[pm][j] = -(fnn.class_arrays[pm][j] - fnn.layers.List2_elem[L].list2_elem[pm][j]) * diff(fnn.layers_in.List2_elem[L - 1].list2_elem[pm][j]);
                        a += bpa.diff_layersin.List2_elem[L - 1].list2_elem[pm][j] * fnn.layers.List2_elem[L - 1].list2_elem[pm][i];
                    }
                    bpa.diff_weight_arrays.List2_elem[L - 1].list2_elem[j][i] = a;
                    //cout << "-------------------" << endl << a << endl;
                }
            }
        }
        else {
            for (int j = 0; j < m[k + 1]; j++) {
                for (int i = 0; i < m[k]; i++) {
                    double a = 0;
                    for (int pm = 0; pm < batch_size; pm++) {
                        double b = 0;
                        for (int p = 0; p < m[k + 2]; p++) {
                            b += bpa.diff_layersin.List2_elem[k + 1].list2_elem[pm][p] * fnn.weight_arrays.List2_elem[k + 1].list2_elem[p][j];
                            //cout << "+++++++++++++++++++++" << endl << b << endl;
                        }
                        bpa.diff_layersin.List2_elem[k].list2_elem[pm][j] = diff(fnn.layers_in.List2_elem[k].list2_elem[pm][j]) * b;
                        a += bpa.diff_layersin.List2_elem[k].list2_elem[pm][j] * fnn.layers.List2_elem[k].list2_elem[pm][i];
                        //cout << "*****************" << endl << a << endl;
                    }
                    bpa.diff_weight_arrays.List2_elem[k].list2_elem[j][i] = a;
                    //cout << "*****************" << endl << a << endl;
                }
            }
        }
    }

    for (int k = L - 1; k >= 0; k--) {
        for (int j = 0; j < m[k + 1]; j++) {
            bpa.diff_bias.List_elem[k].list_elem[j] = bpa.diff_layersin.List2_elem[k].list2_elem[0][j];
            //cout << "*****************" << endl << diff_bias.List_elem[k].list_elem[j] << endl << j << endl;
        }
    }
    //cout << "Function backward is run" << endl;
}
void parameters_update_GD(BPA& bpa, FNN& fnn, double lr) {
    //参数更新：W, b
    int L = bpa.length;
    int batch_size = bpa.batch_size;
    int* m = fnn.nodes;
    //printf("%f\n", lr);

    for (int k = 0; k < L; k++) {
        for (int j = 0; j < m[k + 1]; j++) {
            for (int i = 0; i < m[k]; i++) {
                //printf("weight_arrays_pre: %f\n", fnn.weight_arrays.List2_elem[k].list2_elem[j][i]);
                //printf("diff_weight_arrays: %f\n", bpa.diff_weight_arrays.List2_elem[k].list2_elem[j][i]);
                fnn.weight_arrays.List2_elem[k].list2_elem[j][i] -= lr * bpa.diff_weight_arrays.List2_elem[k].list2_elem[j][i];
                //printf("weight_arrays: %f\n", fnn.weight_arrays.List2_elem[k].list2_elem[j][i]);
            }

            fnn.bias.List_elem[k].list_elem[j] -= lr * bpa.diff_bias.List_elem[k].list_elem[j];
        }
    }
    printf("Optim:GD\n");

    bpa.first = false;
    printf("Function parameters_update_GD is run\n");
}
void parameters_update_Momentum(BPA& bpa, OPT& opt, FNN& fnn, double lr, int iter) {
    //参数更新：W, b
    int L = bpa.length;
    int batch_size = bpa.batch_size;
    int* m = fnn.nodes;

    double beta = 0.9;
    for (int k = 0; k < L; k++) {
        for (int j = 0; j < m[k + 1]; j++) {
            for (int i = 0; i < m[k]; i++) {
                if (opt.first || (iter % update) == 0) {
                    opt.diff_momentum.List2_elem[k].list2_elem[j][i] = (1 - beta) * bpa.diff_weight_arrays.List2_elem[k].list2_elem[j][i];
                    fnn.weight_arrays.List2_elem[k].list2_elem[j][i] -= lr * opt.diff_momentum.List2_elem[k].list2_elem[j][i];
                }
                else {
                    opt.diff_momentum.List2_elem[k].list2_elem[j][i] = beta * opt.diff_momentum.List2_elem[k].list2_elem[j][i] + (1 - beta) * bpa.diff_weight_arrays.List2_elem[k].list2_elem[j][i];
                    fnn.weight_arrays.List2_elem[k].list2_elem[j][i] -= lr * opt.diff_momentum.List2_elem[k].list2_elem[j][i];
                }

            }
            if (opt.first || (iter % update) == 0) {
                opt.diff_momentum_b.List_elem[k].list_elem[j] = (1 - beta) * bpa.diff_bias.List_elem[k].list_elem[j];
                fnn.bias.List_elem[k].list_elem[j] -= lr * opt.diff_momentum_b.List_elem[k].list_elem[j];
            }
            else {
                opt.diff_momentum_b.List_elem[k].list_elem[j] = beta * opt.diff_momentum_b.List_elem[k].list_elem[j] + (1 - beta) * bpa.diff_bias.List_elem[k].list_elem[j];
                fnn.bias.List_elem[k].list_elem[j] -= lr * opt.diff_momentum_b.List_elem[k].list_elem[j];
            }
        }
    }
    printf("Optim:Momentum\n");
    opt.first = false;
    printf("Function parameters_update_momentum is run\n");
}
void parameters_update_Adam(BPA& bpa, OPT& opt, FNN& fnn, double lr, int iter) {
    //参数更新：W, b
    int L = bpa.length;
    int batch_size = bpa.batch_size;
    int* m = fnn.nodes;

    double beta = 0.9; double beta1 = 0.999;
    double v = 0; double u = 0; double ee = 1e-4;
    for (int k = 0; k < L; k++) {
        for (int j = 0; j < m[k + 1]; j++) {
            for (int i = 0; i < m[k]; i++) {
                if (opt.first || (iter % update) == 0) {
                    opt.diff_momentum.List2_elem[k].list2_elem[j][i] = (1 - beta) * bpa.diff_weight_arrays.List2_elem[k].list2_elem[j][i];
                    opt.diff_rmsprop.List2_elem[k].list2_elem[j][i] = (1 - beta1) * pow(bpa.diff_weight_arrays.List2_elem[k].list2_elem[j][i], 2);
                    v = opt.diff_momentum.List2_elem[k].list2_elem[j][i] / (1 - beta);
                    u = opt.diff_rmsprop.List2_elem[k].list2_elem[j][i] / (1 - beta1);
                    fnn.weight_arrays.List2_elem[k].list2_elem[j][i] -= lr * v / (sqrt(u) + ee);
                }
                else {
                    opt.diff_momentum.List2_elem[k].list2_elem[j][i] = beta * opt.diff_momentum.List2_elem[k].list2_elem[j][i] + (1 - beta) * bpa.diff_weight_arrays.List2_elem[k].list2_elem[j][i];
                    opt.diff_rmsprop.List2_elem[k].list2_elem[j][i] = beta1 * opt.diff_rmsprop.List2_elem[k].list2_elem[j][i] + (1 - beta1) * pow(bpa.diff_weight_arrays.List2_elem[k].list2_elem[j][i], 2);
                    v = opt.diff_momentum.List2_elem[k].list2_elem[j][i] / (1 - beta);
                    u = opt.diff_rmsprop.List2_elem[k].list2_elem[j][i] / (1 - beta1);
                    fnn.weight_arrays.List2_elem[k].list2_elem[j][i] -= lr * v / (sqrt(u) + ee);
                }
            }
            if (opt.first || (iter % update) == 0) {
                opt.diff_momentum_b.List_elem[k].list_elem[j] = (1 - beta) * bpa.diff_bias.List_elem[k].list_elem[j];
                opt.diff_rmsprop_b.List_elem[k].list_elem[j] = (1 - beta1) * pow(bpa.diff_bias.List_elem[k].list_elem[j], 2);
                v = opt.diff_momentum_b.List_elem[k].list_elem[j] / (1 - beta);
                u = opt.diff_rmsprop_b.List_elem[k].list_elem[j] / (1 - beta1);
                fnn.bias.List_elem[k].list_elem[j] -= lr * v / (sqrt(u) + ee);
            }
            else {
                opt.diff_momentum_b.List_elem[k].list_elem[j] = beta * opt.diff_momentum_b.List_elem[k].list_elem[j] + (1 - beta) * bpa.diff_bias.List_elem[k].list_elem[j];
                opt.diff_momentum_b.List_elem[k].list_elem[j] = beta1 * opt.diff_momentum_b.List_elem[k].list_elem[j] + (1 - beta1) * pow(bpa.diff_bias.List_elem[k].list_elem[j], 2);
                v = opt.diff_momentum_b.List_elem[k].list_elem[j] / (1 - beta);
                u = opt.diff_rmsprop_b.List_elem[k].list_elem[j] / (1 - beta1);
                fnn.bias.List_elem[k].list_elem[j] -= lr * v / (sqrt(u) + ee);
            }
        }
    }
    printf("Optim:Adam\n");
    opt.first = false;
    printf("Function parameters_update_Adam is run\n");
}

bool read_weight_arrays(List2& weight_arrays, int L) {
    FILE* fin_weight;
    errno_t err_weight = 0;
    err_weight = fopen_s(&fin_weight, "..\\TRAIN\\weight_arrays.txt", "r");
    if (err_weight != 0) {
        printf("Can't read the data of weight_arrays!!!\n");
    }

    for (int k = 0; k < L; k++) {
        for (int i = 0; i < weight_arrays.List2_elem[k].rows; i++) {
            for (int j = 0; j < weight_arrays.List2_elem[k].cols; j++) {
                fscanf_s(fin_weight, "%lf", &weight_arrays.List2_elem[k].list2_elem[i][j]);
            }
        }
    }

    fclose(fin_weight);
    return true;
}
bool read_bias(List& bias, int L) {
    FILE* fin_bias;
    errno_t err_bias = 0;
    err_bias = fopen_s(&fin_bias, "..\\TRAIN\\bias.txt", "r");
    if (err_bias != 0) {
        printf("Can't read the data of bias!!!\n");
    }

    for (int k = 0; k < L; k++) {
        for (int i = 0; i < bias.List_elem[k].list_elem_size; i++) {
            fscanf_s(fin_bias, "%lf", &bias.List_elem[k].list_elem[i]);
            //printf("%lf\n", bias.List_elem[k].list_elem[i]);
        }
    }

    fclose(fin_bias);
    return true;
}
bool read_offset(long* offset) {
    FILE* fin_offset;
    errno_t err_offset = 0;
    err_offset = fopen_s(&fin_offset, "..\\TRAIN\\offset.txt", "r");
    if (err_offset != 0) {
        printf("Can't read the data of offset!!!\n");
    }

    for (int k = 0; k < 2; k++) {
        fscanf_s(fin_offset, "%ld", &offset[k]);
        //printf("%ld\n", offset[k]);
    }

    fclose(fin_offset);
    return true;
}
bool saveDataFromMnist_random(int batch_size) {
    FILE* imgs;
    FILE* labs;
    errno_t err_img = 0; errno_t err_lab = 0;
    err_img = fopen_s(&imgs, add_imgs, "rb");
    err_lab = fopen_s(&labs, add_labs, "rb");
    if (err_img != 0) {
        printf("Can't Open This File: Images!!!\n");
    }
    if (err_img != 0) {
        printf("Can't Open This File: Labels!!!\n");
    }

    int magic;				//文件中的魔术数(magic number)  
    int num_items;			//mnist图像集文件中的图像数目  
    int num_label;			//mnist标签集文件中的标签数目  
    int rows;				//图像的行数  
    int cols;				//图像的列数  

    fread(&magic, sizeof(int), 1, imgs);
    if (swap(magic) != 2051) {
        printf("This isn't the Mnist Images File!!!\n");
    }
    fread(&magic, sizeof(int), 1, labs);
    if (swap(magic) != 2049) {
        printf("This isn't the Mnist Labels File!!!\n");
    }

    fread(&num_items, sizeof(int), 1, imgs);
    fread(&num_label, sizeof(int), 1, labs);
    if (swap(num_items) != swap(num_label)) {
        printf("The Image File and Label File are not a Pair!!!\n");
    }

    fread(&rows, sizeof(int), 1, imgs);
    fread(&cols, sizeof(int), 1, imgs);
    rows = swap(rows); cols = swap(cols);

    int sizes = rows * cols;
    if (sizes != img_size) {
        printf("The Size of Picture is False!!!\n");
    }

    char* pixels_img = new char[sizes];
    int* imgs_random = new int[sizes];
    char label;
    //int c;
    FILE* fout_random_imgs;
    errno_t err_random_imgs = 0;
    err_random_imgs = fopen_s(&fout_random_imgs, "imgs_random.txt", "w");
    FILE* fout_random_labs;
    errno_t err_random_labs = 0;
    err_random_labs = fopen_s(&fout_random_labs, "labs_random.txt", "w");

    double mn;
    for (int i = 0; i < batch_size; i++) {
        int c = rand() % 9 - 1;
        long a = 16 + i * img_size * 10 + c * img_size;
        long b = 8 + i * 10 + c;
        fseek(imgs, a, SEEK_SET);
        fseek(labs, b, SEEK_SET);
        fread(pixels_img, sizeof(char), sizes, imgs);
        fread(&label, sizeof(char), 1, labs);
        //图像数据
        for (int j = 0; j < sizes; j++) {
            mn = pixels_img[j];
            //cout<< m << "--";
            if (mn == 0) {
                fprintf(fout_random_imgs, "%d  ", 0);
                if (i > 4990)
                    printf("%d***", 0);
            }
            else {
                fprintf(fout_random_imgs, "%d  ", 1);
                if (i > 4990)
                    printf("%d***", 1);
            }
        }
        //标签数据
        //printf("++++++++++++++++++++++++++\n");
        int classes_k = (int)label;
        fprintf(fout_random_labs, "%d  ", classes_k);
        printf("%d\n", classes_k);
    }
    delete[] pixels_img;

    fclose(imgs);
    fclose(labs);
    fclose(fout_random_imgs);
    fclose(fout_random_labs);
    return true;
}
bool save_weight_arrays(List2 weight_arrays, int L) {
    FILE* fout_weight;
    errno_t err_weight = 0;
    err_weight = fopen_s(&fout_weight, "weight_arrays.txt", "w");
    if (err_weight != 0) {
        printf("Can't save the data of weight_arrays!!!\n");
    }

    for (int k = 0; k < L; k++) {
        for (int i = 0; i < weight_arrays.List2_elem[k].rows; i++) {
            for (int j = 0; j < weight_arrays.List2_elem[k].cols; j++) {
                fprintf(fout_weight, "%lf  ", weight_arrays.List2_elem[k].list2_elem[i][j]);
            }
        }
    }

    fclose(fout_weight);
    return true;
}
bool save_bias(List bias, int L) {
    FILE* fout_bias;
    errno_t err_bias = 0;
    err_bias = fopen_s(&fout_bias, "bias.txt", "w");
    if (err_bias != 0) {
        printf("Can't save the data of bias!!!\n");
    }

    for (int k = 0; k < L; k++) {
        for (int i = 0; i < bias.List_elem[k].list_elem_size; i++) {
            fprintf(fout_bias, "%lf  ", bias.List_elem[k].list_elem[i]);
        }
    }

    fclose(fout_bias);
    return true;
}
bool save_offset(long* offset) {
    FILE* fout_offset;
    errno_t err_offset = 0;
    err_offset = fopen_s(&fout_offset, "offset.txt", "w");
    if (err_offset != 0) {
        printf("Can't save the data of offset!!!\n");
    }

    for (int k = 0; k < 2; k++) {
        fprintf(fout_offset, "%ld  ", offset[k]);
    }

    fclose(fout_offset);
    return true;
}
void FNNTrain(FNN& fnn, int times, char* optim, int L, int* m, bool first) {
    double lr_GD = 0.01;
    double lr_Momentum = 0.01;
    double lr_Adam = 0.001;
    char Momentum[] = "Momentum";
    char GD[] = "GD";
    char Adam[] = "Adam";
    double loss = 0;
    BPA bpa;
    OPT opt;
    if (strcmp(optim, Momentum) == 0 || strcmp(optim, Adam) == 0) {
        Init_Network_OPT(opt, L, m, batch_sizes);
    }
    Init_Network_FNN(fnn, L, m, batch_sizes);
    Init_Network_BPA(bpa, L, m, batch_sizes);
    long* offset = new long[2];

    if (first) {
        createDatas(fnn.weight_arrays, fnn.bias, L, m);
        //read_offset(offset);
        //offset = readDataFromMnist_train_con(fnn.layers.List2_elem[0].list2_elem, fnn.class_arrays, offset);
        offset = readDataFromMnist_train(fnn.layers.List2_elem[0].list2_elem, fnn.class_arrays);
    }
    else {
        read_weight_arrays(fnn.weight_arrays, L);
        read_bias(fnn.bias, L);
        read_offset(offset);
        offset = readDataFromMnist_train_con(fnn.layers.List2_elem[0].list2_elem, fnn.class_arrays, offset);
    }

    int i = 0; int j = 0; int iter = 0;
    while (i < times) {
        iter++;
        loss = forward(fnn);
        backward(bpa, fnn);
        if (strcmp(optim, GD) == 0) {
            parameters_update_GD(bpa, fnn, lr_GD);
        }
        else if (strcmp(optim, Momentum) == 0) {
            parameters_update_Momentum(bpa, opt, fnn, lr_Momentum, iter);
        }
        else if (strcmp(optim, Adam) == 0) {
            parameters_update_Adam(bpa, opt, fnn, lr_Adam, iter);
        }
        printf("Times: %d\n", i + 1);
        printf("Iteration %d\n**********************\n", j + 1);
        i++;
        if ((loss < 1e-6) || (i == times)) {
            j++;
            if (j >= batch) {
                break;
            }
            offset = readDataFromMnist_train_con(fnn.layers.List2_elem[0].list2_elem, fnn.class_arrays, offset);
            opt.first = true;
            i = 0; iter = 0;
        }
    }

    save_weight_arrays(fnn.weight_arrays, fnn.length - 1);
    save_bias(fnn.bias, fnn.length - 1);
    save_offset(offset);
    delete[] offset;

    memset(&bpa, 0, sizeof(bpa));
    memset(&opt, 0, sizeof(opt));
}
void FNNTrain_Random(FNN& fnn, int times, char* optim, int L, int* m, bool first) {
    double lr_GD = 0.01;
    double lr_Momentum = 0.01;
    double lr_Adam = 0.001;
    char Momentum[] = "Momentum";
    char GD[] = "GD";
    char Adam[] = "Adam";
    double loss = 0;
    BPA bpa;
    OPT opt;
    if (strcmp(optim, Momentum) == 0 || strcmp(optim, Adam) == 0) {
        Init_Network_OPT(opt, L, m, batch_sizes);
    }
    Init_Network_FNN(fnn, L, m, batch_sizes);
    Init_Network_BPA(bpa, L, m, batch_sizes);
    long* offset = new long[2];

    if (first) {
        //createDatas(fnn.weight_arrays, fnn.bias, L, m);
        read_weight_arrays(fnn.weight_arrays, L);
        read_bias(fnn.bias, L);
        //read_offset(offset);
        //offset = readDataFromMnist_random_con(fnn.layers.List2_elem[0].list2_elem, fnn.class_arrays, offset);
        offset = readDataFromMnist_random(fnn.layers.List2_elem[0].list2_elem, fnn.class_arrays);
    }
    else {
        read_weight_arrays(fnn.weight_arrays, L);
        read_bias(fnn.bias, L);
        read_offset(offset);
        offset = readDataFromMnist_random_con(fnn.layers.List2_elem[0].list2_elem, fnn.class_arrays, offset);
    }

    int i = 0; int j = 0; int iter = 0;
    while (i < times) {
        iter++;
        loss = forward(fnn);
        backward(bpa, fnn);
        if (strcmp(optim, GD) == 0) {
            parameters_update_GD(bpa, fnn, lr_GD);
        }
        else if (strcmp(optim, Momentum) == 0) {
            parameters_update_Momentum(bpa, opt, fnn, lr_Momentum, iter);
        }
        else if (strcmp(optim, Adam) == 0) {
            parameters_update_Adam(bpa, opt, fnn, lr_Adam, iter);
        }
        printf("Times: %d\n", i + 1);
        printf("Iteration %d\n**********************\n", j + 1);
        i++;
        if ((loss < 1e-6) || (i == times)) {
            j++;
            if (j >= batch) {
                break;
            }
            offset = readDataFromMnist_random_con(fnn.layers.List2_elem[0].list2_elem, fnn.class_arrays, offset);
            opt.first = true;
            i = 0; iter = 0;
        }
    }

    save_weight_arrays(fnn.weight_arrays, fnn.length - 1);
    save_bias(fnn.bias, fnn.length - 1);
    save_offset(offset);
    delete[] offset;

    memset(&bpa, 0, sizeof(bpa));
    memset(&opt, 0, sizeof(opt));
}

int main() {
    srand((unsigned)time(NULL));
    int L = 3;											    //神经网络的层数
    int m[] = { img_size, 300,50, classes };				//每一层的结点数：m[1:]
    char optimizer[] = "Adam";
    int random_numbers = 5000;
    bool first = true;
    bool random = true;
    //加一行注释
    FNN Net;
    FNNTrain(Net, times, optimizer, L, m, first);
    //saveDataFromMnist_random(random_numbers);
    //FNNTrain_Random(Net, times, optimizer, L, m, first);
    memset(&Net, 0, sizeof(Net));
    return 1;
}