# 对w2v向量矩阵进行奇异值分解

import numpy as np
import cvxpy as cp
from delete_bias import *


def w2v_svd(w2v_model):
    w2v_vectors = w2v_model.wv.vectors
    w2v_vectors_unit = []
    # 将每个向量单位化
    for i in range(116180):
        w2v_vectors_unit.append(unit_word(w2v_vectors[i]))

    W_T = np.array(w2v_vectors_unit)  # 输出矩阵维度是97w*300
    W = W_T.T  # 300*97w
    # print(W.shape)
    U, Z, V = np.linalg.svd(W)
    return U, Z, V


# Define randomized SVD function
def rSVD(w2v_model, r, q=0, p=0):
    w2v_vectors = w2v_model.wv.vectors
    w2v_vectors_unit = []
    # 将每个向量单位化
    for i in range(len(w2v_vectors)):
        w2v_vectors_unit.append(unit_word(w2v_vectors[i]))

    W_T = np.array(w2v_vectors_unit)  # 输出矩阵维度是97w*300
    X = W_T.T  # 300*97w

    # Step 1: Sample column space of X with P matrix
    ny = X.shape[1]
    P = np.random.randn(ny, r + p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z, mode='reduced')

    # Step 2: Compute SVD on projected Y = Q.T @ X
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY

    # print("U")
    # for i in U:
    #     print(i)
    # print("--------------------------")
    # print("S")
    # for i in S:
    #     print(i)

    print("U:" + str(U.shape))
    print("S:" + str(S.shape))

    return U, S, VT


def w2v_sdp(w2v_model):
    # U,Z,V=w2v_svd(w2v_model)
    U, Z, V = rSVD(w2v_model, 50)
    print("--------------------------SVD完成")
    U_T = U.T
    A = Z * U_T
    C = U * Z

    N_T_list = []
    with open("no_sex.txt", encoding='utf-8') as f_n:
        for i in f_n:
            word = i.replace("\n", "")
            try:
                N_T_list.append(unit_word(w2v_model.wv[word]))
            except:
                wwwww = 0
                # print(word+"不存在")
    print("性别中性词数量" + str(len(N_T_list)))
    N_T = np.array(N_T_list)

    B_T = np.array(identify_gender_subspace(w2v_model))
    B = B_T.T  # 300*1

    X = cp.Variable((50, 50))
    term_1 = cp.norm((A @ X - A) @ C, p='fro')
    term_2 = cp.norm(N_T @ (X @ B), p='fro')
    constraints = [X >> 0]
    constraints += [X.T == X]  # 对称
    # 正定
    # constraints += [np.all(np.linalg.eigvals(X) > 0)]

    print("定义完成！")

    prob = cp.Problem(cp.Minimize(term_1 + 0.2 * term_2), constraints)
    prob.solve()

    print("The optimal value is", prob.value)
    print("The optimal x is")
    print(X.value)

    with open("x.txt", "w", encoding='utf-8') as f_x:
        for i in range(50):
            for j in range(50):
                f_x.write(str(X.value[i][j]))
                f_x.write(" ")
            f_x.write("\n")

        f_x.close()
    print("X求解完成！")

    T=np.linalg.cholesky(X.value)
    print(T)
    with open("t.txt", "w", encoding='utf-8') as f_t:
        for i in range(50):
            for j in range(50):
                f_t.write(str(T[i][j]))
                f_t.write(" ")
            f_t.write("\n")

        f_t.close()
    print("T求解完成！")


def sdp_test():
    # Generate a random SDP.
    n = 3
    p = 3
    np.random.seed(1)
    C = np.random.randn(n, n)
    A = []
    b = []
    for i in range(p):
        A.append(np.random.randn(n, n))
        b.append(np.random.randn())

    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((n, n), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    constraints += [
        cp.trace(A[i] @ X) == b[i] for i in range(p)
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),
                      constraints)
    prob.solve()

    # Print result.
    print("The optimal value is", prob.value)
    print("A solution X is")
    print(X.value)

# def sdp_x_print_test():
#     X=np.random.random((50,50))
#
#     with open("x.txt", "w", encoding='utf-8') as f_x:
#         for i in range(50):
#             for j in range(50):
#                 f_x.write(str(X[i][j]))
#                 f_x.write(" ")
#             f_x.write("\n")
#
#         f_x.close()
#     print("X求解完成！")