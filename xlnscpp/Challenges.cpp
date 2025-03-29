#define xlns_ideal

#include "xlns32.cpp"
#include "xlns16.cpp"
#include <iostream>

using namespace std;


float** fp(float** A, int m, int p, float** B, int n) {
    float** C = new float*[m];
    for (int i = 0; i < m; i++) {
        C[i] = new float[n];
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0f;
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < p; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}


float** transpose(float** mat, int m, int n) {
    float** result = new float*[n];
    for (int i = 0; i < n; i++) {
        result[i] = new float[m];
    }
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

xlns32_float** matMul_xlns32(xlns32_float** A, int m, int p, xlns32_float** B, int n) {
    xlns32_float** C = new xlns32_float*[m];
    for (int i = 0; i < m; i++) {
        C[i] = new xlns32_float[n];
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
        }
    }
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            for (int k = 0; k < p; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

xlns32_float** transpose(xlns32_float** mat, int m, int n) {
    xlns32_float** result = new xlns32_float*[n];
    for (int i = 0; i < n; i++) {
        result[i] = new xlns32_float[m];
    }
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            result[j][i] = mat[i][j];
        }
    }
    return result;
}


xlns16_float** matMul_xlns16(xlns16_float** A, int m, int p, xlns16_float** B, int n) {
    xlns16_float** C = new xlns16_float*[m];
    for (int i = 0; i < m; i++) {
        C[i] = new xlns16_float[n];
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
        }
    }
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            for (int k = 0; k < p; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

xlns16_float** transpose(xlns16_float** mat, int m, int n) {
    xlns16_float** result = new xlns16_float*[n];
    for (int i = 0; i < n; i++) {
        result[i] = new xlns16_float[m];
    }
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            result[j][i] = mat[i][j];
        }
    }
    return result;
}


template<typename T>
void printMat(T** mat, int m, int n) {
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}


template<typename T>
void freeMat(T** mat, int m) {
    for (int i = 0; i < m; i++){
        delete[] mat[i];
    }
    delete[] mat;
}

int main() {
    

    int m = 4, p = 2, n = 3;
    
    float** A_fp = new float*[m];
    A_fp[0] = new float[p]{2.0f, 8.0f};
    A_fp[1] = new float[p]{5.0f, 1.0f};
    A_fp[2] = new float[p]{4.0f, 2.0f};
    A_fp[3] = new float[p]{8.0f, 6.0f};
    

    float** B_fp = new float*[n];
    B_fp[0] = new float[p]{10.0f, 5.0f};
    B_fp[1] = new float[p]{9.0f, 9.0f};
    B_fp[2] = new float[p]{5.0f, 4.0f};
    
    
    float** B_fp_T = transpose(B_fp, n, p);
    float** result_fp = fp(A_fp, m, p, B_fp_T, n);
    cout << "Result float matrix:" << endl;
    printMat(result_fp, m, n);
    


    xlns32_float** A_x32 = new xlns32_float*[m];
    A_x32[0] = new xlns32_float[p];
    A_x32[1] = new xlns32_float[p];
    A_x32[2] = new xlns32_float[p];
    A_x32[3] = new xlns32_float[p];

    for(int i = 0;i<m;i++){
        for(int j = 0;j<p;j++){
            A_x32[i][j] = A_fp[i][j];
        }
    }

    
    xlns32_float** B_x32 = new xlns32_float*[n];
    B_x32[0] = new xlns32_float[p];
    B_x32[1] = new xlns32_float[p];
    B_x32[2] = new xlns32_float[p];
    
    for(int i = 0;i<n;i++){
        for(int j = 0;j<p;j++){
            B_x32[i][j] = B_fp[i][j];
        }
    }
    
    xlns32_float** B_x32_T = transpose(B_x32, n, p);
    
    xlns32_float** result_x32 = matMul_xlns32(A_x32, m, p, B_x32_T, n);
    cout << "Result xlns32_float matrix:" << endl;
    printMat(result_x32, m, n);
    

    xlns16_float** A_x16 = new xlns16_float*[m];
    A_x16[0] = new xlns16_float[p];
    A_x16[1] = new xlns16_float[p];
    A_x16[2] = new xlns16_float[p];
    A_x16[3] = new xlns16_float[p];

    for(int i = 0;i<m;i++){
        for(int j = 0;j<p;j++){
            A_x16[i][j] = A_fp[i][j];
        }
    }
    

    xlns16_float** B_x16 = new xlns16_float*[n];
    B_x16[0] = new xlns16_float[p];
    B_x16[1] = new xlns16_float[p];
    B_x16[2] = new xlns16_float[p];

    for(int i = 0;i<n;i++){
        for(int j = 0;j<p;j++){
            B_x16[i][j] = B_fp[i][j];
        }
    }
    
    xlns16_float** B_x16_T = transpose(B_x16, n, p);
    
    xlns16_float** result_x16 = matMul_xlns16(A_x16, m, p, B_x16_T, n);
    cout << "Result xlns16_float matrix:" << endl;
    printMat(result_x16, m, n);
    
    
    // Free the space
    freeMat(A_fp, m);
    freeMat(B_fp, n);
    freeMat(B_fp_T, p);
    freeMat(result_fp, m);
    
    freeMat(A_x32, m);
    freeMat(B_x32, n);
    freeMat(B_x32_T, p);
    freeMat(result_x32, m);
    
    freeMat(A_x16, m);
    freeMat(B_x16, n);
    freeMat(B_x16_T, p);
    freeMat(result_x16, m);
    
    return 0;
}
