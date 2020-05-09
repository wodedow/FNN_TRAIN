#ifndef STRUCT_H
#define STRUCT_H
struct list {
	double* list_elem;
	int list_elem_size;
};

struct List {
	list* List_elem;
	int list_size;
};

struct list2 {
	double** list2_elem;
	int cols;
	int rows;
};

struct List2 {
	list2* List2_elem;
	int list2_size;
};

void InitList_elem(list& List_elem, int initListElemSize) {
	//��ʼ�����Ա�
	List_elem.list_elem = new double[initListElemSize];
	List_elem.list_elem_size = initListElemSize;
}

void InitList(List& LList, int initListSize) {
	//��ʼ�����Ա�
	LList.List_elem = new list[initListSize];
	LList.list_size = initListSize;
}

void InitList2_elem(list2& List2_elem, int list2_rows, int list2_cols) {
	//��ʼ�����Ա�
	List2_elem.list2_elem = new double* [list2_rows];
	for (int i = 0; i < list2_rows; i++) {
		List2_elem.list2_elem[i] = new double[list2_cols];
	}
	List2_elem.cols = list2_cols;
	List2_elem.rows = list2_rows;
}

void InitList2(List2& LList2, int List2_Size) {
	//��ʼ�����Ա�
	LList2.List2_elem = new list2[List2_Size];
	LList2.list2_size = List2_Size;
}

/*************************************************************************************/
struct FNN {
	bool first;
	int length;
	int batch_size;
	int* nodes;
	double loss;							//��ʧֵ $ E_{z}=\sum_{p^m=0}\frac{1}{2}\sum_{l=0}(t_{l}^{L,p^m}-x_{l}^{L,p^m})^2 $
	//char regulation[] = "L2";				//Ĭ��ȡL2����
	List bias;								//ƫ�ã�$ b $
	List2 layers_in;						//�����������
	List2 layers;							//����������������ʼֵ��
	List2 weight_arrays;					//Ȩ�ؾ���$ W $
	double** class_arrays;					//��ʼ���ݵ����
};
void Init_Network_FNN(FNN& Net, int L, int* m, int batch_size) {
	Net.length = L + 1;
	Net.batch_size = batch_size;
	Net.nodes = m;
	InitList(Net.bias, L);
	InitList2(Net.layers, L + 1);
	InitList2(Net.weight_arrays, L);
	InitList2(Net.layers_in, L);
	//��ʼ�����ݽṹ��
	for (int i = 1; i <= L + 1; i++) {
		InitList2_elem(Net.layers.List2_elem[i - 1], batch_size, m[i - 1]);
		if (i == L + 1) {
			break;
		}
		InitList2_elem(Net.weight_arrays.List2_elem[i - 1], m[i], m[i - 1]);
		InitList2_elem(Net.layers_in.List2_elem[i - 1], batch_size, m[i]);
		InitList_elem(Net.bias.List_elem[i - 1], m[i]);
	}
	//��ʼ�����
	double** a = new double* [batch_size];
	for (int ii = 0; ii < batch_size; ii++) {
		a[ii] = new double[m[L]];
	}
	Net.class_arrays = a;
}

/*************************************************************************************/
struct BPA {
	bool first;
	int length;
	int batch_size;
	List diff_bias;						//�洢��ƫ�õĵ������� $ \frac{\partial E_{p^{m}}}{b_j} $
	List2 diff_layersin;				//�洢���������ݵĵ������� $ \frac{\partial E_{p^{m}}}{\partial u_j^{k+1,p^m}} $
	List2 diff_weight_arrays;			//�洢��ӦȨ�صĵ������� $ \frac{\partial E_{p^{m}}}{W_{j,i}} $
};
void Init_Network_BPA(BPA& Alo, int L, int* m, int batch_size) {
	Alo.first = true;
	Alo.length = L;
	Alo.batch_size = batch_size;
	InitList2(Alo.diff_weight_arrays, L);
	InitList2(Alo.diff_layersin, L);
	InitList(Alo.diff_bias, L);

	for (int i = 1; i < L + 1; i++) {
		InitList2_elem(Alo.diff_layersin.List2_elem[i - 1], batch_size, m[i]);
		InitList2_elem(Alo.diff_weight_arrays.List2_elem[i - 1], m[i], m[i - 1]);
		InitList_elem(Alo.diff_bias.List_elem[i - 1], m[i]);
	}
}

/*************************************************************************************/
struct OPT {
	bool first;
	List diff_momentum_b;
	List2 diff_momentum;				//�洢Momentum�е��ۻ��ݶ� $  $
	List diff_rmsprop_b;
	List2 diff_rmsprop;
};
void Init_Network_OPT(OPT& Opt, int L, int* m, int batch_size) {
	Opt.first = true;
	InitList2(Opt.diff_momentum, L);
	InitList(Opt.diff_momentum_b, L);
	InitList2(Opt.diff_rmsprop, L);
	InitList(Opt.diff_rmsprop_b, L);
	for (int i = 1; i < L + 1; i++) {
		InitList2_elem(Opt.diff_momentum.List2_elem[i - 1], m[i], m[i - 1]);
		InitList_elem(Opt.diff_momentum_b.List_elem[i - 1], m[i]);
		InitList2_elem(Opt.diff_rmsprop.List2_elem[i - 1], m[i], m[i - 1]);
		InitList_elem(Opt.diff_rmsprop_b.List_elem[i - 1], m[i]);
	}
}

/*************************************************************************************/
struct TES {
	bool first;
	List layers_test;
};
void Init_Network_TES(TES& Tes, int L, int* m) {
	Tes.first = true;
	InitList(Tes.layers_test, L + 1);
	for (int i = 1; i <= L + 1; i++) {
		InitList_elem(Tes.layers_test.List_elem[i - 1], m[i - 1]);
	}
}

#pragma once
#endif