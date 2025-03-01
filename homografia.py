# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nomes: Dionnatas Brito & Meilen Salamanca
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv

#########################################################################################################################
#
# Retirado do exercise_homography02.ipynb
def normalize_points(points):
    """
    Normaliza os pontos para melhorar a estabilidade numérica no cálculo da homografia.
    Returns:
        - Matriz de transformação de normalização 3x3.
        - Pontos normalizados em coordenadas homogêneas.
    """
    centroid = np.mean(points[0:2, :], axis=1)
    distance = np.linalg.norm(points[0:2, :] - centroid.reshape(2, 1), axis=0)
    avg_distance = np.mean(distance)
    if avg_distance == 0:
        scale = 1 
    else:
        scale = np.sqrt(2) / avg_distance
    # scale = np.sqrt(2) / avg_distance #Aqui estava com erro dando a divisão por zero e pegando poucos inliers, então fiz a verificação antes

    # Define a matriz de normalização
    T = np.array([[scale, 0, -scale * centroid[0]], 
                  [0, scale, -scale * centroid[1]], 
                    [0, 0, 1]])
    
    # Aplica a normalização
    norm_pts = T @ points
    return T, norm_pts

#########################################################################################################################
#
# Retirado do exercise_homography02.ipynb
def my_dlt(pts1, pts2):
    """
    Calcula a matriz de homografia utilizando o algoritmo Direct Linear Transform (DLT).
    Returns:
        - Matriz de homografia.
    """
    npoints = pts1.shape[1]
    A = np.zeros((3 * npoints, 9))

    for k in range(npoints):
        A[3 * k, 3:6] = -pts2[2, k] * pts1[:, k]
        A[3 * k, 6:9] = pts2[1, k] * pts1[:, k]

        A[3 * k + 1, 0:3] = pts2[2, k] * pts1[:, k]
        A[3 * k + 1, 6:9] = -pts2[0, k] * pts1[:, k]

        A[3 * k + 2, 0:3] = -pts2[1, k] * pts1[:, k]
        A[3 * k + 2, 3:6] = pts2[0, k] * pts1[:, k]

    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_matrix = np.reshape(h, (3, 3))

    return H_matrix

#########################################################################################################################
#
# Retirado do exercise_homography02.ipynb
def my_homography(pts1, pts2):
    """
    Calcula a matriz de homografia normalizada a partir de correspondências de pontos.
    Returns:
        - Matriz de homografia 3x3.
    """
    # Normaliza os pontos
    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T
    pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1)))).T

    T1, norm_pts1 = normalize_points(pts1)
    T2, norm_pts2 = normalize_points(pts2)

    #Caso tenha a divisão por zero no normalize_points, irá exibir uma figura com a mensagem de erro
    if np.any(np.isnan(norm_pts1)) or np.any(np.isnan(norm_pts2)):
        plt.figure()
        plt.text(0.5, 0.5, "Erro: Todos os pontos são idênticos\nou há poucos pontos distintos.\n\n\n LOG: Olhe para a figura de 'Pontos Detectados por RANSAC'\nNote que há poucos 'Inliers'\nVocê deve rodar novamente o código para obter um resultado válido.",
                fontsize=12, ha='center', va='center')
        plt.axis("off") 
        plt.show()

    # Aplica o DLT
    H = my_dlt(norm_pts1, norm_pts2)

    # Denormaliza a homografia
    H = np.linalg.inv(T2) @ H @ T1
    return H
########################################################################################################################
#
#Função auxiliar que calcula o erro de reprojeção
def compute_reprojection_error(H, pts1, pts2):
    """
    Calcula o erro de reprojeção entre os pontos transformados pela homografia e os pontos de destino.
    Returns:
        - Erro médio de reprojeção.
    """
    num_points = pts1.shape[0]
    pts1_hom = np.concatenate([pts1, np.ones((num_points, 1))], axis=1).T
    pts1_proj = (H @ pts1_hom).T
    pts1_proj /= pts1_proj[:, 2][:, np.newaxis]  # Normalización
    
    dists = np.linalg.norm(pts1_proj[:, :2] - pts2, axis=1)
    return np.mean(dists)

########################################################################################################################
#
#Função que implementa o RANSAC
def ransac(pts1, pts2, dis_threshold, p, e, s, N):
    """
    Implementa o algoritmo RANSAC para estimar a melhor matriz de homografia.
    Returns:
        - Matriz de homografia.
        - Pontos inliers da imagem de origem.
        - Pontos inliers da imagem de destino.
        - Lista de erros por iteração.
        - Erro final de reprojeção.
    """
    max_inliers = 0
    H_best = None
    inliers_best = np.zeros(pts1.shape[0], dtype=bool)
    num_points = pts1.shape[0]
    sample_count = 0
    errors_per_iteration = []
    print("####################################################################################################")
    print("LOG - Iniciando RANSAC")
    print("LOG - Parametros:")
    while sample_count < N:
        indices = random.sample(range(num_points), s)
        sample_pts1 = pts1[indices].reshape(s, 2)
        sample_pts2 = pts2[indices].reshape(s, 2)
        
        H = my_homography(sample_pts1, sample_pts2)
        
        if H is None:
            sample_count += 1
            continue
        
        
        error = compute_reprojection_error(H, pts1, pts2)
        errors_per_iteration.append(error)
        
        pts1_hom = np.concatenate([pts1, np.ones((num_points, 1))], axis=1).T
        pts1_proj = (H @ pts1_hom).T
        pts1_proj /= pts1_proj[:, 2][:, np.newaxis]
        
        dists = np.linalg.norm(pts1_proj[:, :2] - pts2, axis=1)
        inliers = dists < dis_threshold
        num_inliers = np.sum(inliers)

        print(f"LOG - Interacao {sample_count + 1}/{N} - Numero de inliers: {num_inliers} - Melhor ate agora: {max_inliers}/{num_points} - e: {e:.6f} - N: {N}")

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            H_best = H
            inliers_best = inliers.copy()
            e = min(max(1 - (num_inliers / num_points), 1e-6), 0.99)
            print("  ####################################################################################################")
            print("  LOG - Novo melhor erro encontrado")
            print(f"  LOG - e:  {e}")
            #e = 1 - (num_inliers / num_points)

            if e == 0:
                break
            N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** 4)))
            print("  LOG - Atualizando N")
            print(f"  LOG - N:  {N}")
            print("  ####################################################################################################")
            if num_inliers > 0:
                N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** 4)))

        
        sample_count += 1
    inlier_pts1 = pts1[inliers_best]
    inlier_pts2 = pts2[inliers_best]

    plt.scatter(pts1[:, 0], pts1[:, 1], c='red', label='Outliers')
    plt.scatter(inlier_pts1[:, 0], inlier_pts1[:, 1], c='green', label='Inliers')
    plt.legend()
    plt.title("Pontos detectados por RANSAC")
    plt.show(block=False)

    
    if H_best is not None:
            H_best = my_homography(pts1[inliers_best], pts2[inliers_best])

    final_error = compute_reprojection_error(H_best, pts1[inliers_best], pts2[inliers_best]) if H_best is not None else None
    

    print("   LOG -        RESULTADO FINAL:      ")
    print("   LOG - Numero de iteracoes:", sample_count)
    print("   LOG - Melhor erro:", final_error)
    print("   LOG - Numero de inliers:", max_inliers, "de", num_points,)
    print(f"   LOG - Inliers:{(np.sum(max_inliers)/num_points)*100:.2f}%")
    print("  ####################################################################################################")

    return H_best, pts1[inliers_best], pts2[inliers_best], errors_per_iteration, final_error
########################################################################################################################
def are_points_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    for x, y in points[2:]:
        if abs((y2 - y1) * (x - x1) - (y - y1) * (x2 - x1)) > 1e-6:
            return False
    return True


########################################################################################################################
#
# Fornecido pela professora
# Exemplo de Teste da função de homografia usando o SIFT

MIN_MATCH_COUNT = 10
#img1 = cv.imread('./assets/img/270_left.jpg', 0)   # queryImage
#img2 = cv.imread('./assets/img/270_right.jpg', 0)        # trainImage

img2 = cv.imread('./assets/img/batman.jpg', 0)   # queryImage
img1 = cv.imread('./assets/img/outdoor_batman.jpg', 0) 

# img2 = cv.imread('./assets/img/monalisa01_1.jpg', 0)   # queryImage
# img1 = cv.imread('./assets/img/monalisa02.jpg', 0) 


# img2 = cv.imread('./assets/img/outdoors02.jpg', 0)   # queryImage
# img1 = cv.imread('./assets/img/outdoors01.jpg', 0) 

# Inicialização do SIFT
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
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    # Ajustar parámetros de RANSAC
    p = 0.99
    e = 0.2  
    #dis_threshold = 5.0  
    dis_threshold = 3 #np.max([0.02*(np.sqrt(img2.shape[0]**2+img2.shape[1]**2)),2])
    s = 8  
    N = 2000  

    H_RANSAC, src_in, dst_in, errors_per_iteration, final_error = ransac(
        src_pts.reshape(-1, 2), dst_pts.reshape(-1, 2), dis_threshold, p, e, s, N
)
    
    if H_RANSAC is not None:
        img4 = cv.warpPerspective(img1, H_RANSAC, (img2.shape[1], img2.shape[0]))
    else:
        img4 = img1  # Define um
    
else:
    print(" LOG - Nao se pode realizar homografia - {}/{}".format(len(good), MIN_MATCH_COUNT))
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
plt.title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################