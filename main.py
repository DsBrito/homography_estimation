# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: 

# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv

#########################################################################################################################
#
# Retirado do exercise_homography02.ipynb
def my_dlt(pts1,pts2):
    """
        Estima a matriz de homografia utilizando o algoritmo Direct Linear Transform (DLT).

        Parâmetros:
        pts1 : Conjunto de N pontos da primeira imagem, representados em coordenadas homogêneas.
        pts2 : Conjunto de N pontos correspondentes na segunda imagem, também em coordenadas homogêneas.

        Retorno:
        H_matrix : Matriz de homografia estimada que transforma os pontos de pts1 para pts2.
    """
    # Compute matrix A
    npoints = pts1.shape[1]
    A = np.zeros((3*npoints, 9))

    for k in range(npoints):
        A[3*k,3:6] = -pts2[2,k]*pts1[:,k]
        A[3*k,6:9] = pts2[1,k]*pts1[:,k]

        A[3*k+1,0:3] = pts2[2,k]*pts1[:,k]
        A[3*k+1,6:9] = -pts2[0,k]*pts1[:,k]

        A[3*k+2,0:3] = -pts2[1,k]*pts1[:,k]
        A[3*k+2,3:6] = pts2[0,k]*pts1[:,k]

    U,S,Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_matrix = np.reshape(h,(3,3))

    return H_matrix
#
########################################################################################################################
#
#Retirado do exercise_homography02.ipynb
def my_homography(pts1,pts2):
    """
    Estima a matriz de homografia entre dois conjuntos de pontos correspondentes,
    aplicando normalização para melhor precisão numérica e utilizando o algoritmo DLT.

    Parâmetros:
    pts1 : Conjunto de N pontos da primeira imagem, representados em coordenadas cartesianas (x, y).
    pts2 : Conjunto de N pontos correspondentes na segunda imagem, também em coordenadas cartesianas.

    Retorno:
    H : Matriz de homografia estimada que transforma os pontos de pts1 para pts2.
    """
    # Normalize points => homogenia
    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T
    pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1)))).T

    T1, norm_pts1 = normalize_points(pts1)
    T2, norm_pts2 = normalize_points(pts2)

    # Perform DLT and obtain normalized matrix
    H = my_dlt(norm_pts1,norm_pts2)

    # Denormalize the homography matrix
    H = np.linalg.inv(T2) @ H @ T1

    return H
########################################################################################################################
#
#Retirado do exercise_homography02.ipynb
def normalize_points(points):
    """
    Normaliza um conjunto de pontos em coordenadas homogêneas para melhorar a estabilidade numérica 
    em cálculos geométricos, como na estimação de homografias.

    Parâmetros:
    points :  Matriz contendo N pontos em coordenadas homogêneas (x, y, 1), organizados em colunas.

    Retorno:
    T : Matriz de transformação que normaliza os pontos através de uma translação para o centróide 
        e um escalonamento para uma média de distância √2.
    norm_pts : Conjunto de pontos normalizados obtidos após a aplicação da matriz de transformação T.
    """
    centroid =np.mean (points[0:2,:], axis=1)
    # print("centroid:",centroid.reshape(2,1))

    # Calculate the average distance of the points having the centroid as origin
    distance =np.linalg.norm(points[0:2,:] - centroid.reshape(2,1), axis=0)
    # print("distance:",distance)

    avg_distance = np.mean(distance)
    # print("avg_distance:",avg_distance)

    scale = np.sqrt(2) / avg_distance
    # print("scale",scale)

    # Define the normalization matrix (similar transformation)
    T =  np.array([[scale, 0, -scale*centroid[0]], [0, scale, -scale*centroid[1]], [0,0,1]])
    # print("T:",T)

    norm_pts  =  T @ points
    # print("norm_pts:",norm_pts)

    return T, norm_pts
#
########################################################################################################################
def ransac(pts1,pts2,dis_threshold,p,e,s,N):
    """
    Implementa o algoritmo RANSAC para estimar uma matriz de homografia robusta,
    lidando com correspondências ruidosas ou outliers nos pontos de entrada.

    Parâmetros:
    pts1 : Conjunto de pontos da primeira imagem.
    pts2 : Conjunto de pontos correspondentes na segunda imagem.
    dis_threshold : Limite de erro para considerar um ponto como inlier.
    p : Probabilidade desejada de encontrar um modelo livre de outliers.
    e : Proporção estimada de outliers nos dados.
    s : Número mínimo de pontos necessários para estimar a homografia (normalmente 4 ou 8).
    N : Número máximo de iterações permitidas.

    Retorno:
    H : Matriz de homografia estimada.
    pts1_in : Conjunto de pontos inliers da primeira imagem.
    pts2_in :  Conjunto de pontos inliers da segunda imagem.
    """        
    #determinação dos parametros do RANSAC
    pts1 = np.squeeze(pts1)
    pts2 = np.squeeze(pts2)

# Agora você pode concatenar com np.ones
    pts2_homo = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis=1)
    pts1_homo = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis=1)

    # Transformando os pontos em homogeneos
    #pts2_homo = np.concatenate((pts2,np.ones((pts2.shape[0],1))),axis=1)
    #pts1_homo = np.concatenate((pts1,np.ones((pts1.shape[0],1))),axis=1)
    
    #valor de T
    T = round(pts1.shape[0]*(1-e))
    print("RANSAC (parametros): Maximo de interacoes: ",int(N),"|| Limiar T (numero de inliers) = ",T)
    
    inliers = np.array([])
    Hfinal = None
    
    # melhor erro
    best_error = np.inf    
      
    # para um subset 
    for n in range(0,round(N)):
    
        # Passo 1 - Selecione aleatoriamente um c
        # onjunto de s amostras
    
        # criar uma lista de index aleatório
        list = random.sample(range(0,len(pts1)), s)
        
        # Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2 
        # Vetor que vai receber os 8 pontos aleatórios
        pts1_random = np.zeros((len(list),pts1.shape[1]))
        pts2_random = np.zeros((len(list),pts2.shape[1]))
        i = 0
        
        # os vetores receberão os valores aleatórios
        for indice in list:
            pts1_random[i,:] = pts1[indice,:]
            pts2_random[i,:] = pts2[indice,:]
            i += 1
                    
            
        # Usa as amostras para estimar uma homografia usando o DTL Normalizado
        H = my_homography(pts1_random, pts2_random)
        
        # Passo 3 - Calcular o erro associado
        # Erro de transferência simetrico
        
        #calcular os pontos preditos pelo modelo
        pts1_predicted = np.dot(np.linalg.inv(H),pts2_homo.T).T

        #deixar o ultimo valor igual a 1
        pts1_predicted[:,0]/= pts1_predicted[:,-1] 
        pts1_predicted[:,1]/= pts1_predicted[:,-1] 
        pts1_predicted[:,2]/= pts1_predicted[:,-1] 
            
        pts2_predicted = np.dot(H,pts1_homo.T).T
        pts2_predicted[:,0]/= pts2_predicted[:,-1] 
        pts2_predicted[:,1]/= pts2_predicted[:,-1] 
        pts2_predicted[:,2]/= pts2_predicted[:,-1]

        
        # simetrica (perda e custo total)
        src_loss = np.linalg.norm(pts1_predicted - pts1_homo, axis=1)
        dst_loss = np.linalg.norm(pts2_predicted - pts2_homo, axis=1)
        cost = src_loss + dst_loss
        
    
        # numero de amostras que estão abaixo do erro maximo
        new_inliers =  np.where(cost<dis_threshold)[0]
        
        #caso o numero de inliers aumente atualiza os inliers e H
        if new_inliers.shape[0]>inliers.shape[0]:
            inliers = new_inliers.copy()
            best_error = np.median(cost)
        
        
        
        # verificar os critérios de parada        

        
        if inliers.shape[0] >= T:
            print("Quantidade de inliers minimos atingidos= ", inliers.shape[0])
            break

    print("RANSAC (resultado): Numero de épocas:",n+1,"|| Melhor erro:", best_error,
            "|| Número de inliers:", inliers.shape[0],'\n')
    
    # Terminado o processo iterativo
    # Estima a homografia final H usando todos os inliers selecionados.

    H = my_homography(pts1[inliers,:], pts2[inliers,:])
    pts1_in = pts1[inliers,:]
    pts2_in = pts2[inliers,:]
    return H, pts1_in, pts2_in
#
#
########################################################################################################################

# Exemplo de Teste da função de homografia usando o SIFT


MIN_MATCH_COUNT = 10
img1 = cv.imread('./assets/img/270_left.jpg', 0)   # queryImage
img2 = cv.imread('./assets/img/270_right.jpg', 0)        # trainImage

# Inicialização do SIFT
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
    
    # =============================================================================
    # RANSAC
    # =============================================================================
    p = 0.99
    e = 0.2 # (80%)

    dis_threshold = np.max([0.01*(np.sqrt(img2.shape[0]**2+img2.shape[1]**2)),1])
    
    # Conjunto de amostras
    s = 8
    # N
    # N = 10000 

    N = 1024 

    H_RANSAC,src_in,dst_in = ransac(src_pts,dst_pts,dis_threshold,p,e,s,N)
    M = H_RANSAC

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 3)
plt.title('Analisando os pontos correspondentes')
plt.imshow(img3, 'gray')

fig.add_subplot(2, 2, 1)
plt.title('Primeira imagem')
plt.imshow(img1, 'gray')

fig.add_subplot(2, 2, 2)
plt.title('Segunda imagem')
plt.imshow(img2, 'gray')

fig.add_subplot(2, 2, 4)
plt.title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################
