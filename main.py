from KryuGL import *

# Posicion de la Camara
eye = V3(1, 1, 5)
center = V3(0, 0, 0)
displace = V3(0, 1, 0)

# Llama las funciones para la escritura de la imagen .bmp
r = Bitmap(600, 600)
r.glViewPort(1, 1, 599, 599)
r.lookAt(eye, center, displace)

# Se elige un shader a utilizar en los modelos siguientes
r.shader = r.gouraud

# ---------- Cargar los modelos bmp con textura mtl ----------
# X-Wing
r.glLoad(
    './Modelos/x.obj',
    mtl='./Modelos/x.mtl',
    translate=(1.25,0.5,0),
    scale=(0.39,0.39,0.39),
    rotate=(0.2,0,-0.06)
    )
    
r.glClearZbuffer()

# ------------------------------------------------------------

# Creacion del Archivo
r.glFinish('Materials_15146.bmp')
