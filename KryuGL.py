"""
    -------------------------------------------------------------------------
    Universidad del Valle de Guatemala
    Ingenieria en Ciencias de la Computacion y Tecnologias de la Informacion
    Graficas por Computadora - Seccion 10
    Dieter de Wit - Carnet 15146
    -------------------------------------------------------------------------
    Funciones de Creacion de objetos Bitmap, lectura de ellos y mimica de GL
    -------------------------------------------------------------------------
"""

# **************** Imports ***************
import struct
from collections import namedtuple
import math
# ****************************************


# --- Estructurar datos String como Binarios ---
def char(c):
    return struct.pack("=c", c.encode('ascii'))


def word(c):
    return struct.pack("=h", c)


def dword(c):
    return struct.pack("=l", c)
# ----------------------------------------------


# Formato de Color RGB
def color(r, g, b):
    return bytes([b, g, r])


# -------------------------------- Operaciones sobre Vectores ----------------------------------
V2 = namedtuple('Vertex2', ['x', 'y'])
V3 = namedtuple('Vertex3', ['x', 'y', 'z'])


def suma(v0, v1):
    return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)


def sub(v0, v1):
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)


def mul(v0, k):
    return V3(v0.x * k, v0.y * k, v0.z * k)


def dot(v0, v1):
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z


def cross(v0, v1):
    return V3(v0.y * v1.z - v0.z * v1.y, v0.z * v1.x - v0.x * v1.z, v0.x * v1.y - v0.y * v1.x)


def length(v0):
    return (v0.x ** 2 + v0.y ** 2 + v0.z ** 2) ** 0.5


def norm(v0):
    nlength = length(v0)
    if nlength == 0:
        return V3(0, 0, 0)
    else:
        return V3(v0.x / nlength, v0.y / nlength, v0.z / nlength)
# ----------------------------------------------------------------------------------------------


# Muestreo de los tamanos de recuadros
def bbox(A, B, C):
    xs = sorted([A.x, B.x, C.x])
    ys = sorted([A.y, B.y, C.y])
    v1 = V2(int(xs[0]), int(ys[0]))
    v2 = V2(int(xs[2]), int(ys[2]))
    return v1, v2


# Transformacion de coordenadas de los vertices
def barycentric(A, B, C, P):
    cx, cy, cz = cross(
        V3(B.x - A.x, C.x - A.x, A.x - P.x),
        V3(B.y - A.y, C.y - A.y, A.y - P.y)
    )
    if cz == 0: # cz no puede ser menor que 1
        return -1, -1, -1

    # Se calculan las coordenadas baricentricas
    u = cx/cz
    v = cy/cz
    w = 1 - (u + v)
    return w, v, u


# Funcion de Multiplicacion de 2 Matrices
def mult_matrices(matrix1, matrix2):
    resultante = []
    filas = len(matrix1)
    columnas = len(matrix2[0])
    longitud = len(matrix1[0])
    longitud2 = len(matrix2)

    if longitud != longitud2:
        return None

    for i in range(filas):
        fila = []
        for j in range(columnas):
            count = 0
            for k in range(longitud):
                count += matrix1[i][k] * matrix2[k][j]
            fila.append(count)
        resultante.append(fila)
    return resultante


# ************************************************************************************
# Objeto: Permite la escritura a un archivo BMP y funciones de Renderizado
# Bitmap
# ************************************************************************************
class Bitmap(object):

    # Inicializacion General: Variables
    def __init__(self, width, height):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.width = width
        self.height = height
        self.framebuffer = []
        self.zbuffer = []
        self.shader = color(0, 0, 0)
        self.material = {}
        self.vertex_color = color(255, 255, 0)
        self.colorin = self.shader
        self.Model = []
        self.View = []
        self.Projection = []
        self.Viewport = []
        self.glInit()

    # Inicializacion Secundaria: Funciones que Inicializan nuestro Software Renderer
    def glInit(self):
        self.glClear()

    # Crecion del Viewport que tiene las dimensiones de Escritura del BMP
    def glViewPort(self, x, y, width, height):
        self.vx = width
        self.vy = height
        self.x = x
        self.y = y

    # Realiza la limpieza de ambos Buffers
    def glClear(self):
        self.framebuffer = [
            [
                color(0, 0, 0)  # Escritura de pixeles negros
                for x in range(self.width)
            ]
            for y in range(self.height)
        ]

        self.zbuffer = [
            [
                -1 * float('inf')
                for x in range(self.width)
            ]
            for y in range(self.height)
        ]

    # Limpiaeza del zbuffer para escritura de nuevos .obj
    def glClearZbuffer(self):
        self.zbuffer = [
            [
                -1 * float('inf')
                for x in range(self.width)
            ]
            for y in range(self.height)
        ]

    # Funcion de escritura de un pixel
    def glVertex(self, x, y):
        try:
            self.framebuffer[y][x] = self.vertex_color
        except IndexError:
            pass

    # Varia el color del glVertex
    def glColor(self, r, g, b):
        self.vertex_color = color(r, g, b)

    # Creacion del archivo .bmp
    def glFinish(self, filename):
        f = open(filename, 'bw')

        # File Header (14)
        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(162 + self.width * self.height))
        f.write(dword(0))
        f.write(dword(54))

        # Image Header (40)
        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))

        # Escritura al Buffer
        for x in range(self.height):
            for y in range(self.width):
                f.write(self.framebuffer[x][y])

        # Se cierra el archivo
        f.close()

    # Funcion de linea en cualquier cuadrante
    def glLine(self, x0, y0, x1, y1):

        # Diferenciales para la pendiente
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        # Si la diferencia de y es mayor se debe de pintar mas pixeles en ese eje
        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        # Cuando el punto inicial en x es mayor que el final se cambian
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        # Calculamos las diferenciales por si hubo cambio de puntos
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        offset = 0
        threshold = dx
        y = y0

        for x in range(x0, x1 + 1):
            if steep:
                self.glVertex(y, x)
            else:
                self.glVertex(x, y)

            offset += dy
            if offset >= threshold:
                y += 1 if y0 < y1 else -1
                threshold += dx

    # z-buffer
    def glTransform(self, vertex, translate=(0, 0, 0), scale=(1, 1, 1)):
        # Vertex aumentado para la Matriz Pipe resultante
        matrix_vertex = [[vertex[0]], [vertex[1]], [vertex[2]], [1]]

        # Transformar un vertice a Tupla
        matriz1 = mult_matrices(self.Model, self.View)
        matriz2 = mult_matrices(self.Projection, matriz1)
        matriz3 = mult_matrices(self.Viewport, matriz2)
        transformed = mult_matrices(matriz3, matrix_vertex)

        # Se construye el vertice ya transformado
        vtransformed = [
            (transformed[0][0] / transformed[3][0]),
            (transformed[1][0] / transformed[3][0]),
            (transformed[2][0] / transformed[3][0])
        ]
        return V3(
            (vtransformed[0] + translate[0]) * scale[0],
            (vtransformed[1] + translate[1]) * scale[1],
            (vtransformed[2] + translate[2]) * scale[2]
        )

    # Lee y pinta la imagen de fondo por medio de un framebuffer
    def framebuffer_fondo(self, texture):
        for x in range(texture.width):
            for y in range(texture.height):
                fcolor = texture.pixels[y][x]
                self.glColor(fcolor[0], fcolor[1], fcolor[2])
                self.glVertex(x, y)

    # Funcion de camara que define desde que angulo realizamos la escritura de nuestro modelos
    def lookAt(self, eye, center, up):
        # Coordenadas de Proyeccion
        z = norm(sub(eye, center))
        x = norm(cross(up, z))
        y = norm(cross(z, x))

        # Creacion de la matriz View
        self.loadViewMatrix(x, y, z, center)
        self.loadProjectionMatrix(-1 / length(sub(eye, center)))
        self.loadViewportMatrix()

    # Creacion y uso de las Matrices de Traslacion, Rotacion y Escala a utilizar
    def loadModelMatrix(self, translate=(0, 0, 0), scale=(1, 1, 1), rotate=(0, 0, 0)):
        # Creacion de la matriz Translate
        mtranslate = [
            [1, 0, 0, translate[0]],
            [0, 1, 0, translate[1]],
            [0, 0, 1, translate[2]],
            [0, 0, 0, 1]
        ]
        # Creacion de la matriz Scale
        mscale = [
            [scale[0], 0, 0, 0],
            [0, scale[1], 0, 0],
            [0, 0, scale[2], 0],
            [0, 0, 0, 1]
        ]

        # ---- Creacion de las tres matrices Rotation ----
        a = rotate[0]
        rotation_x = [
            [1, 0, 0, 0],
            [0, math.cos(a), -1 * (math.sin(a)), 0],
            [0, math.sin(a), math.cos(a), 0],
            [0, 0, 0, 1]
        ]
        a = rotate[1]
        rotation_y = [
            [math.cos(a), 0, math.sin(a), 0],
            [0, 1, 0, 0],
            [-1 * (math.sin(a)), 0, math.cos(a), 0],
            [0, 0, 0, 1]
        ]
        a = rotate[2]
        rotation_z = [
            [math.cos(a), -1 * (math.sin(a)), 0, 0],
            [math.sin(a), math.cos(a), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        # ------------------------------------------------

        r_matrix = mult_matrices(rotation_y, rotation_z)
        r_matrix = mult_matrices(r_matrix, rotation_x)
        matrix = mult_matrices(r_matrix, mscale)
        self.Model = mult_matrices(mtranslate, matrix)

    def loadViewMatrix(self, x, y, z, center):
        M = [
            [x.x, x.y, x.z, 0],
            [y.x, y.y, y.z, 0],
            [z.x, z.y, z.z, 0],
            [0, 0, 0, 1]
        ]
        O = [
            [1, 0, 0, -center.x],
            [0, 1, 0, -center.y],
            [0, 0, 1, -center.z],
            [0, 0, 0, 1]
        ]
        self.View = mult_matrices(M, O)

    def loadProjectionMatrix(self, coeff):
        self.Projection = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, coeff, 1]
        ]

    def loadViewportMatrix(self, x=0, y=0):
        w = int(self.width / 2)
        h = int(self.height / 2)
        self.Viewport = [
            [w, 0, 0, x + w],
            [0, w, 0, y + h],
            [0, 0, 128, 128],
            [0, 0, 0, 1]
        ]

    def glTriangle(self, A, B, C, directional_light, normal_coords, mtl_color=None):
        bbox_min, bbox_max = bbox(A, B, C)

        for x in range(bbox_min.x, bbox_max.x + 1):
            for y in range(bbox_min.y, bbox_max.y + 1):
                w, v, u = barycentric(A, B, C, V2(x, y))

                if w < 0 or v < 0 or u < 0:
                    continue

                self.colorin = self.shader(bar = (w, v, u), varying_normals = normal_coords, light = directional_light, mtl_color = mtl_color, xyCoords = (x, y))

                self.glColor(self.colorin[0], self.colorin[1], self.colorin[2])

                z = A.z * w + B.z * v + C.z * u

                if x < 0 or y < 0:
                    continue

                if x < len(self.zbuffer) and y < len(self.zbuffer[x]) and z > self.zbuffer[x][y]:
                    self.glVertex(x, y)
                    self.zbuffer[x][y] = z

    # Carga el Obj y realiza las operaciones de Traslacion, Escala, Rotacion y aplicacion del Material
    def glLoad(self, filename, translate=(0, 0, 0), scale=(1, 1, 1), rotate=(0, 0, 0), mtl='None'):
        self.loadModelMatrix(translate, scale, rotate)
        model = Obj(filename)
        self.glRead(mtl)

        for face in model.faces:
            vcount = len(face)

            if vcount >= 3:
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1

                a = self.glTransform(model.vertices[f1], translate, scale)
                b = self.glTransform(model.vertices[f2], translate, scale)
                c = self.glTransform(model.vertices[f3], translate, scale)

                n1 = face[0][2] - 1
                n2 = face[1][2] - 1
                n3 = face[2][2] - 1
                nA = V3(*model.normales[n1])
                nB = V3(*model.normales[n2])
                nC = V3(*model.normales[n3])

                luz = V3(0.1, 0.6, 0.9)

                red = round(self.material[face[3]][0] * 255)
                green = round(self.material[face[3]][1] * 255)
                blue = round(self.material[face[3]][2] * 255)

                self.glTriangle(A=a, B=b, C=c, directional_light=luz, normal_coords=(nA, nB, nC), mtl_color=(red, green, blue))

    # Lectura del archivo .mtl para el muestreo de los colores del material
    def glRead(self, mtl):
        with open(mtl) as f2:
            lineas = f2.read().splitlines()

        for linea in lineas:
            if linea:
                prefix, value = linea.split(' ', 1)
            if prefix == 'newmtl':
                indice = lineas.index(linea)
                for read_line in range(indice, indice + 5):
                    prefix2, colores = lineas[read_line].split(' ', 1)
                    if prefix2 == 'Kd':
                        self.material.update({value: list(map(float, (colores.split(' '))))})

    # Gouraud Shader que se utiliza para agregar efectos de sombreado asi como aumento a las normales poligonales
    def gouraud(self, **kwargs):

        # Recibe las coordenadas normales, barycentricas, los colores del archivo mtl y la luz
        w, v, u = kwargs['bar']
        nA, nC, nB = kwargs['varying_normals']
        light = kwargs['light']
        r, g, b = kwargs['mtl_color']

        nx = nA.x * w + nB.x * v + nC.x * u
        ny = nA.y * w + nB.y * v + nC.y * u
        nz = nA.z * w + nB.z * v + nC.z * u

        # Se obtiene un vector normal por cada punto
        vertex_normal = V3(nx, ny, nz)

        # Se calcula el valor de la intensidad de la luz
        intensity = dot(vertex_normal, light)

        r = int(r * intensity)
        g = int(g * intensity)
        b = int(b * intensity)

        return (
            r if 255 > r > 0 else 0,
            g if 255 > g > 0 else 0,
            b if 255 > b > 0 else 0
        )

# *************************************************************************************************
# Objeto: Permite la lectura de un archivo .obj para la aplicacion del modelo al Software Renderer
# Obj
# *************************************************************************************************
class Obj(object):
    def __init__(self, filename):
        with open(filename) as f:
            self.lines = f.read().splitlines()

        self.vertices = []
        self.tvertices = []
        self.normales = []
        self.mtl = []
        self.faces = []
        self.lista = []
        self.read()

    def space(self, face):

        self.lista = face.split('/')

        if "" in self.lista:
            self.lista[1] = 0

        return map(int, self.lista)

    def read(self):
        for line in self.lines:
            if line:
                prefix, value = line.split(' ', 1)

                if prefix == 'usemtl':
                    material = value
                    # self.mtl.append(value.split(' '))

                if prefix == 'v':
                    self.vertices.append(list(map(float, value.split(' '))))

                if prefix == 'vt':
                    self.tvertices.append(list(map(float, value.split(' '))))

                elif prefix == 'f':
                    listita = [list(self.space(face)) for face in value.split(' ')]
                    # self.faces.append([list(self.try_int(face)) for face in value.split(' ')])
                    listita.append(material)
                    self.faces.append(listita)
                    # self.faces.append([list(map(int,face.split('/'))) for face in value.split(' ')])

                elif prefix == 'vn':
                    self.normales.append(list(map(float, value.split(' '))))
