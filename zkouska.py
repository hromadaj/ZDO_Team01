import sys
import numpy as np
from scipy import ndimage, signal, misc
import skimage.data, skimage.io, skimage.feature
import matplotlib.pyplot as plt
from skimage.filters import gaussian as gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.io import imread
import skimage.morphology
from skimage.filters import sobel, roberts
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
from matplotlib import cm
from numpy.ma.extras import unique
import json
import math



#Funkce
#############################
def objev_primky(image): 
  # Classic straight-line Hough transform
  # Set a precision of 0.5 degree.
  tested_angles = np.linspace(-np.pi , np.pi, 360, endpoint=False)
  h, theta, d = hough_line(image, theta=tested_angles)

  # Generating figure 1
  '''
  fig, axes = plt.subplots(1, 3, figsize=(15, 6))
  ax = axes.ravel()

  ax[0].imshow(image, cmap=cm.gray)
  ax[0].set_title('Input image')
  ax[0].set_axis_off()
  '''

  angle_step = 0.5 * np.diff(theta).mean()
  d_step = 0.5 * np.diff(d).mean()
  bounds = [np.rad2deg(theta[0] - angle_step),
            np.rad2deg(theta[-1] + angle_step),
            d[-1] + d_step, d[0] - d_step]
  #Vykresleni transformace
  '''
  ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
  ax[1].set_title('Hough transform')
  ax[1].set_xlabel('Angles (degrees)')
  ax[1].set_ylabel('Distance (pixels)')
  ax[1].axis('image')
  '''
  #Vykresleni nalezenych primek
  '''
  ax[2].imshow(image, cmap=cm.gray)
  ax[2].set_ylim((image.shape[0], 0))
  ax[2].set_axis_off()
  ax[2].set_title('Detected lines')
  '''
  #Zadefuinovani vystypni tabulky
  tabulka_linii = [];
  iterace = 0
  for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
      (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
      #Zde se vykresluje
      #ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2),color='r')
      iterace = iterace + 1
      pocatek = [x0,y0]
      konec = [0,-x0*np.tan(angle + np.pi/2) + y0]
      if np.tan(angle + np.pi/2) == 0:
        konec = [10,pocatek[1]]
      elif np.tan(angle) > 100:
        konec = [pocatek[0],10]
      #ID;Pocatek;Konec,Uhel
      tabulka_linii.append([iterace,pocatek,konec,(angle)*180/np.pi])

  return(tabulka_linii)


#Funkce pro odstraneni duplikatnich pruseciku
def odstran_duplikatni_pruseciky(pruseciky):
  for row in pruseciky:
    pruseciky_copy = pruseciky.copy()
    pruseciky_copy.remove(row)
    for i in pruseciky_copy:
      if int(i[2].y) == int(row[2].y) and int(i[2].x) == int(row[2].x):
        pruseciky.remove(i)

#Funkce pro kontrolu zdali podezrely prusecik lezi na objektu
def zkontroluj_pruseciky(pruseciky,obr):
  nove_prus = [];
  for row in pruseciky:
    if 0 < int(row[2].x) < obr.shape[1] and 0 < int(row[2].y) < obr.shape[0]:
      #print("Jsem uvnitr oibrazku")
      #print([row[2].y,row[2].x])
      if obr[int(row[2].y)][int(row[2].x)] == True:
        #print("Netrefa")
        #print([row[2].x,row[2].y])
        #print("----------")
        nove_prus.append(row)
  return nove_prus

#Pomocna funkce pro hledani pruseciku na linich
def lineLineIntersection_moje(primka_A,primka_B):
  A = Point(primka_A[1][0],primka_A[1][1])
  B = Point(primka_A[2][0],primka_A[2][1])
  C = Point(primka_B[1][0],primka_B[1][1])
  D = Point(primka_B[2][0],primka_B[2][1])
  #Zavola se iintersect
  podezrely_prusecik = lineLineIntersection(A, B, C, D)
  return [primka_A[0],primka_B[0],podezrely_prusecik]


#Prevzata funkce pro hledani pruseciku
class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	# Method used to display X and Y coordinates
	# of a point
	def displayPoint(self, p):
		print(f"({p.x}, {p.y})")


def lineLineIntersection(A, B, C, D):
  #Nalezeni ID primek s minimalni odchylkou
  # Line AB represented as a1x + b1y = c1
	a1 = B.y - A.y
	b1 = A.x - B.x
	c1 = a1*(A.x) + b1*(A.y)
	# Line CD represented as a2x + b2y = c2
	a2 = D.y - C.y
	b2 = C.x - D.x
	c2 = a2*(C.x) + b2*(C.y)
	determinant = a1*b2 - a2*b1

	if (determinant == 0):
		# The lines are parallel. This is simplified
		# by returning a pair of FLT_MAX
		return Point(10**9, 10**9)
	else:
		x = (b2*c1 - b1*c2)/determinant
		y = (a1*c2 - a2*c1)/determinant
		return Point(x, y)


def NajdiNearPrimky(primka,tabulka,h,w):
  blizke_primky =  []
  #h = 150
  #w = 20
  for bod in primka:
    for row in tabulka:
      for i in row:
        if (bod[0]-w < i[0] < bod[0]+w and bod[1]-h < i[1] < bod[1]+h):
          blizke_primky.append(row)
          break
  nalezene = []
  blizke_primky.insert(0,primka)
  if len(blizke_primky)>1:
    unique_pairs = list(set(tuple(map(tuple, sublist)) for sublist in blizke_primky))
    converted_list = [list(map(list, pair)) for pair in unique_pairs]
    for i in converted_list:
      nalezene.append(i)
  else:
    nalezene.append(primka)
  #print(nalezene)
  return nalezene

def jsem_v_tabulce(primka,tabulka):
  for bod in primka:
    for i in tabulka:
      for j in i:
        bod_b = np.array(j)
        bod_a = np.array(bod) 
        if np.array_equal(bod_a, bod_b):
          return True
  return False



def process(rezy,stehy,img_og,image_filename,visualize):
  incisions = np.array(rezy,dtype=object)
  incision_alphas = []
  incision_lines = []
  for incision in incisions:
    for (p_1, p_2) in zip(incision[:-1],incision[1:]):
      p1 = np.array(p_1)
      p2 = np.array(p_2)
      dx = p2[0]-p1[0]
      dy = p2[1]-p1[1]
      if dy == 0:
        alpha = 90.0
      elif dx == 0:
        alpha = 0.0
      else:
        alpha = 90 + 180.*np.arctan(dy/dx)/np.pi
      incision_alphas.append(alpha)
      incision_lines.append([p1, p2])

  stitches = np.array(stehy,dtype=object)
  stitche_alphas = []
  stitche_lines = []
  for stitche in stitches:
    for (p_1, p_2) in zip(stitche[:-1],stitche[1:]):
      p1 = np.array(p_1)
      p2 = np.array(p_2)
      dx = p2[0]-p1[0]
      dy = p2[1]-p1[1]
      if dy == 0:
        alpha = 90.0
      elif dx == 0:
        alpha = 180.0
      else:
        alpha = 90 + 180.*np.arctan(dy/dx)/np.pi        
      stitche_alphas.append(alpha)
      stitche_lines.append([p1, p2])



  # analyze alpha for each pair of line segments
  intersections = []
  intersections_alphas = []
  for (incision_line, incision_alpha) in zip(incision_lines, incision_alphas):
    for (stitche_line, stitche_alpha) in zip(stitche_lines, stitche_alphas):

      p0, p1 = incision_line
      pA, pB = stitche_line
      (xi, yi, valid, r, s) = intersectLines(p0, p1, pA, pB)
      if valid == 1:
        intersections.append([xi, yi])
        alpha_diff = abs(incision_alpha - stitche_alpha)
        alpha_diff = 180.0 - alpha_diff if alpha_diff > 90.0 else alpha_diff
        alpha_diff = 90 - alpha_diff
        intersections_alphas.append(alpha_diff)

  if visualize == True:
    plt.figure(figsize=[15,10])
    plt.imshow(img_og)
    plt.title(image_filename,fontsize=20)

    for p_i in incisions:
        p = np.array(p_i)
        plt.plot(p[:,0], p[:,1])

    for p_s in stitches:
        p = np.array(p_s)
        plt.plot(p[:,0], p[:,1])


    for ((xi,yi), alpha) in zip(intersections, intersections_alphas):

        plt.plot(xi, yi, 'o')
        plt.text(xi, yi,'{:2.1f}'.format(alpha), c='green', bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 1}, size='large')

    plt.show()
  
  return intersections, intersections_alphas



def intersectLines( pt1, pt2, ptA, ptB ):
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;   x2, y2 = pt2
    dx1 = x2 - x1;  dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;   xB, yB = ptB;
    dx = xB - x;  dy = yB - y;

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

    # now, the determinant should be OK
    DETinv = 1.0/DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

    # return the average of the two descriptions
    xi = (x1 + r*dx1 + x + s*dx)/2.0
    yi = (y1 + r*dy1 + y + s*dy)/2.0

    ##############
    #found is intersection (xi,yi) in inner segment
    valid = 0
    if x1 != x2:
        if x1 < x2:
            a = x1
            b = x2
        else:
            a = x2
            b = x1
        c = xi
    else:
        #predpoklad, ze pak y jsou ruzne
        if y1 < y2:
            a = y1
            b = y2
        else:
            a = y2
            b = y1
        c = yi
    if (c > a) and (c < b):
        #now second segment
        if x != xB:
            if x < xB:
                a = x
                b = xB
            else:
                a = xB
                b = x
            c = xi
        else:
            #predpoklad, ze pak y jsou ruzne
            if y < yB:
                a = y
                b = yB
            else:
                a = yB
                b = y
            c = yi
        if (c > a) and (c < b):
            valid = 1

    return ( xi, yi, valid, r, s )


##############################################################
#Hlavni funkce pro spusteni programu
def main(img,img_og,image_filename,visualize):
    #prdani Gausovskeho filtru
    filtrace = gaussian_filter(img, sigma=1,channel_axis=None)
    img = filtrace

    #segmentace
    elevation_map = skimage.filters.roberts(img)

    #Nalezeni Otsu
    otsu = threshold_otsu(img)

    #print("---------------------------")
    #print("Segmentuji, prosim vyckejte")
    #print("---------------------------")
    markers = np.zeros_like(img)

    obsazeni = 100
    iterace = 0
    krok = 0.005
    while obsazeni > 10:
        markers_pom = np.zeros_like(img)
        markers_pom[img < (otsu-iterace)] = 2
        iterace = iterace + krok
        znacka = markers_pom == 2
        suma = 0
        for i in range(znacka.shape[1]-1):
            for j in range(znacka.shape[0]-1):
                if znacka[j,i] == True:
                    suma=suma+1
        kontrola = suma/(znacka.shape[1]*znacka.shape[0])*100
        obsazeni = kontrola

    markers[img < (otsu-(iterace-krok))] = 2


    obsazeni = 100
    iterace = 0
    krok = 0.005
    while obsazeni > 60:
        markers_pom = np.zeros_like(img)
        markers_pom[img > (otsu+iterace)] = 1
        iterace = iterace + krok
        znacka = markers_pom == 1
        suma = 0
        for i in range(znacka.shape[1]-1):
            for j in range(znacka.shape[0]-1):
                if znacka[j,i] == True:
                    suma=suma+1
        kontrola = suma/(znacka.shape[1]*znacka.shape[0])*100
        #print(kontrola)
        obsazeni = kontrola

    markers[img > (otsu+(iterace-krok))] = 1

    #Tady se dela vyplneni tech hran
    segmentation = skimage.segmentation.watershed(elevation_map,markers)


    #vykresleni segmentace
    '''
    plt.figure(figsize=[15,10])
    plt.subplot(1,2,1)
    plt.imshow(markers)

    plt.subplot(1,2,2)
    plt.imshow(segmentation == 2)
    plt.show()
    '''

    #metoda se snazi zaplnit bazenky a zapnlujou to dokud nenarazi na hrany

    #Vytvoreni skeletu
    #A aplikace hranoveho detektoru pro Oba smery
    vstup_skelet = segmentation
    skelet = skimage.morphology.skeletonize(vstup_skelet == 2)

    im = skelet;

    sx = ndimage.sobel(im, axis=0, mode='nearest')
    sy = ndimage.sobel(im, axis=1, mode='nearest')
    sob = np.hypot(sx, sy)

    #Vykresleni hranoveho detektoru   
    '''
    plt.figure(figsize=[15,10])
    plt.subplot(1,2,1)
    plt.imshow(sx,cmap='gray')

    plt.subplot(1,2,2)
    plt.imshow(sy,cmap='gray')
    plt.show()
    '''

    # Labelovnai oblasti v jednom smeru (x)
    vstup_label = sx
    velikost_objektu = img.shape[1]/3

    imlabel = label(vstup_label, background=0)
    #plt.imshow(imlabel, cmap='gray');
    np.unique(imlabel)

    velky = 0;
    vsechny = 0;
    objekty_sx = [];

    for i in range(np.max(imlabel)+1):
        #Vykresleni
        if i > 0:
            #print(np.count_nonzero(imlabel == i))
            if np.count_nonzero(imlabel == i) > velikost_objektu:
                # Nahrani objektu do pole
                objekty_sx.append(imlabel == i)
                velky = velky + 1
            else:
                vsechny = vsechny + 1
    #print('POCET VSECH OBJEKTU (x) JE: ' + str(velky+vsechny))
    #print('POCET nalezenych OBJEKTU (x) JE: ' + str(velky))



    # Labelovnai oblasti v druhem smeru (y)
    vstup_label = sy
    velikost_objektu = img.shape[0]/4

    imlabel = label(vstup_label, background=0)
    #plt.imshow(imlabel, cmap='gray');
    np.unique(imlabel)

    velky = 0;
    vsechny = 0;
    objekty_sy = [];

    for i in range(np.max(imlabel)+1):
        #Vykresleni
        if i > 0:
            #print(np.count_nonzero(imlabel == i))
            if np.count_nonzero(imlabel == i) > velikost_objektu:
                # Nahrani objektu do pole
                objekty_sy.append(imlabel == i)
                velky = velky + 1
            else:
                vsechny = vsechny + 1
    #print('POCET VSECH OBJEKTU (y) JE: ' + str(velky+vsechny))
    #print('POCET nalezenych OBJEKTU (y) JE: ' + str(velky))

    #-----------------------------------------------------------------
    #-----------------------------------------------------------------
    #Hledani vsech stehu
    stehy = []
    for objekt in objekty_sy:
        #Nejprve se udela Hough transformace
        tabulka_linii = objev_primky(objekt)

        #Honzovo algoritmus
        #Prekopirovani tabuly linii
        tabulka = tabulka_linii.copy()
        vstupni_obr = objekt
        seznam_pruseciku = [];
        seznam_pod_pruseciku = [];

        for row in tabulka:
            tabulka_copy = tabulka.copy()
            tabulka_copy.remove(row)
            for i in tabulka_copy:
                seznam_pod_pruseciku.append(lineLineIntersection_moje(row,i))

        #Kontrola a odstraneni duplikatu
        #++++++++++++++++++++++++++++++++++++++++++++++++++++
        pruseciky_zk = zkontroluj_pruseciky(seznam_pod_pruseciku,vstupni_obr)
        odstran_duplikatni_pruseciky(pruseciky_zk)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++


        #pridani prvniho a posledniho bodu 
        #POZOR NA RAZENI PODLE OS ZDE jE X
        #++++++++++++++++++++++++++++++++++++++++++++++++++++

        novy_im = np.zeros(vstupni_obr.shape)
        y,x = novy_im.shape;
        list_hodnot= [];
        for i in range(y):
            for j in range(x):
                if vstupni_obr[i,j] == 1:
                    list_hodnot.append([j,i])

        #POZOR [0 = y,1 = x]
        list_hodnot.sort(key = lambda i: i[1])


        #Zkouska aproximace prvni a posledni hodnoty
        r = 20
        pom_min_x = 0
        pom_min_y = 0
        for i in range(r):
            pom_min_x = pom_min_x+list_hodnot[i][1]
            pom_min_y = pom_min_y+list_hodnot[i][0]

        pom_max_x = 0
        pom_max_y = 0
        for i in range(len(list_hodnot)-r,len(list_hodnot)):
            pom_max_x = pom_max_x+list_hodnot[i][1]
            pom_max_y = pom_max_y+list_hodnot[i][0]

        minimalni = [pom_min_y/r,pom_min_x/r]
        maximalni = [pom_max_y/r,pom_max_x/r]

        #minimalni = list_hodnot[0]
        #maximalni = list_hodnot[-1]

        sour_prvni = Point(minimalni[0],minimalni[1]);
        sour_posledni = Point(maximalni[0],maximalni[1]);
        #++++++++++++++++++++++++++++++++++++++++++++++++++++

        #Prekopirovani pruseciku do noveho pole
        #++++++++++++++++++++++++++++++++++++++++++++++++++++
        pruseciky = [];
        for i in pruseciky_zk:
            pruseciky.append([i[2].x,i[2].y])

        #Pridani prvniho aposledniho bodu
        pruseciky.insert(0,[minimalni[0],minimalni[1]])
        pruseciky.append([maximalni[0],maximalni[1]])
        #++++++++++++++++++++++++++++++++++++++++++++++++++++

        #serazeni Opet [opyor na poradi pro stehy je razeni podle X]
        pruseciky.sort(key = lambda i: i[1])
        #++++++++++++++++++++++++++++++++++++++++++++++++++++

        #Pridani pruseciku do pole stehu
        #++++++++++++++++++++++++++++++++++++++++++++++++++++
        stehy.append(pruseciky)

    #hledani vsech rezu
    rezy = []
    for objekt in objekty_sx:
        #Nejprve se udela Hough transformace
        tabulka_linii = objev_primky(objekt)

        #Honzovo algoritmus
        #Prekopirovani tabuly linii
        tabulka = tabulka_linii.copy()
        vstupni_obr = objekt
        seznam_pruseciku = [];
        seznam_pod_pruseciku = [];

        for row in tabulka:
            tabulka_copy = tabulka.copy()
            tabulka_copy.remove(row)
            for i in tabulka_copy:
                seznam_pod_pruseciku.append(lineLineIntersection_moje(row,i))

        #Kontrola a odstraneni duplikatu
        #++++++++++++++++++++++++++++++++++++++++++++++++++++
        pruseciky_zk = zkontroluj_pruseciky(seznam_pod_pruseciku,vstupni_obr)
        odstran_duplikatni_pruseciky(pruseciky_zk)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++


        #pridani prvniho a posledniho bodu 
        #POZOR NA RAZENI PODLE OS ZDE jE Y
        #++++++++++++++++++++++++++++++++++++++++++++++++++++

        novy_im = np.zeros(vstupni_obr.shape)
        y,x = novy_im.shape;
        list_hodnot= [];
        for i in range(y):
            for j in range(x):
                if vstupni_obr[i,j] == 1:
                    list_hodnot.append([j,i])
        
        #POZOR [0 = y,1 = x]
        list_hodnot.sort(key = lambda i: i[0])


        #Zkouska aproximace prvni a posledni hodnoty
        r = 10
        pom_min_x = 0
        pom_min_y = 0
        for i in range(r):
            pom_min_x = pom_min_x+list_hodnot[i][1]
            pom_min_y = pom_min_y+list_hodnot[i][0]

        pom_max_x = 0
        pom_max_y = 0
        for i in range(len(list_hodnot)-r,len(list_hodnot)):
            pom_max_x = pom_max_x+list_hodnot[i][1]
            pom_max_y = pom_max_y+list_hodnot[i][0]

        minimalni = [pom_min_y/r,pom_min_x/r]
        maximalni = [pom_max_y/r,pom_max_x/r]

        #minimalni = list_hodnot[0]
        #maximalni = list_hodnot[-1]

        sour_prvni = Point(minimalni[0],minimalni[1]);
        sour_posledni = Point(maximalni[0],maximalni[1]);
        #++++++++++++++++++++++++++++++++++++++++++++++++++++

        #Prekopirovani pruseciku do noveho pole
        #++++++++++++++++++++++++++++++++++++++++++++++++++++
        pruseciky = [];
        for i in pruseciky_zk:
            pruseciky.append([i[2].x,i[2].y])

        #Pridani prvniho aposledniho bodu
        pruseciky.insert(0,[minimalni[0],minimalni[1]])
        pruseciky.append([maximalni[0],maximalni[1]])
        #++++++++++++++++++++++++++++++++++++++++++++++++++++

        #serazeni Opet [opyor na poradi pro rezy je razeni podle Y]
        pruseciky.sort(key = lambda i: i[0])
        #++++++++++++++++++++++++++++++++++++++++++++++++++++

        #Pridani pruseciku do pole rezu
        #++++++++++++++++++++++++++++++++++++++++++++++++++++
        rezy.append(pruseciky)



    #Vykresleni vysledku bez spojeni linii
    '''
    plt.figure(figsize=[15,10])
    plt.subplot(1,2,1)
    plt.imshow(img_og);



    
    #Vykresleni rezu
    iterace = 0
    for primka in rezy:
    #primka.sort(key = lambda i: i[0])
        for i in primka:
            plt.scatter(i[0],i[1],color="b")
        x = [point[0] for point in primka]
        y = [point[1] for point in primka]
        plt.plot(x,y,'b');


    #Vykresleni stehu
    for primka in stehy:
        for i in primka:
            plt.scatter(i[0],i[1],color="r")
        x = [point[0] for point in primka]
        y = [point[1] for point in primka]
        plt.plot(x,y,'r');
    '''

    #Spojeni stehu
    #-------------------------------------------
    stehy_deleted = stehy.copy()
    final_stehy = []
    for i in range(len(stehy_deleted)):
        temp = []
        primka = stehy_deleted.pop(0)
        kontrola = jsem_v_tabulce(primka,final_stehy) 
        if kontrola == True:
            continue
        else:
            temp = NajdiNearPrimky(primka,stehy_deleted,30,10)
            #print(len(temp))
            if len(temp)>=1:
                #mergne body
                temp = [item for sublist in temp for item in sublist]
                #serad body
                temp.sort(key = lambda i: i[1])
            #else:
                #serad body
                #temp.sort(key = lambda i: i[1])
            final_stehy.append(temp)

    #Spojeni rezu
    #-------------------------------------------
    rezy_deleted = rezy.copy()
    final_rezy = []
    #print(len(rezy))
    if len(rezy) > 1: 
        for i in range(len(rezy_deleted)):
            temp = []
            primka = rezy_deleted.pop(0)
            kontrola = jsem_v_tabulce(primka,final_rezy) 
            if kontrola == True:
                continue
            else:
                temp = NajdiNearPrimky(primka,rezy_deleted,10,10)
                #print(len(temp))
                if len(temp)>=1:
                    #mergne body
                    temp = [item for sublist in temp for item in sublist]
                #serad body
                #temp.sort(key = lambda i: i[1])
                #else:
                #serad body
                temp.sort(key = lambda i: i[0])
                final_rezy.append(temp)
    else:
        final_rezy = rezy.copy() 

    '''
    #Vykresleni vysledku po spojeni linii
    
    plt.figure(figsize=[15,10])
    #plt.subplot(1,2,2)
    plt.imshow(img_og);
    plt.title(image_filename, fontsize=20)

    #Vykresleni rezu
    iterace = 0
    for primka in final_rezy:
        #primka.sort(key = lambda i: i[0])
        for i in primka:
            plt.scatter(i[0],i[1],color="b")
        x = [point[0] for point in primka]
        y = [point[1] for point in primka]
        plt.plot(x,y,'b');


    #Vykresleni stehu
    for primka in final_stehy:
        for i in primka:
            plt.scatter(i[0],i[1],color="r")
        x = [point[0] for point in primka]
        y = [point[1] for point in primka]
        plt.plot(x,y,'r'); 
    '''


    intersections,angles = process(final_rezy,final_stehy,img_og,image_filename,visualize)

    #for i in intersections:
    #    plt.scatter(i[0],i[1],color="y")
    #plt.show()
    

    #Zapis do json souboru
    data_list = []
    data={}
    data['filename'] = image_filename
    #data['width'] = img_og.shape[1]
    #data['height'] = img_og.shape[0]
    data['incision_polyline'] = []
    data['stitches_polyline'] = []
    for steh in final_stehy:
        data['stitches_polyline'].append(steh)
    for rez in final_rezy:
        data['incision_polyline'].append(rez)
    data['crossing_possitions'] = []
    for i in intersections:
        data['crossing_possitions'].append(i)
    data['crossing_angles'] = []
    for i in angles:
        data['crossing_angles'].append(i)


    data_list.append(data)

    return data_list



#################################################################
#################################################################

#Program
#nacteni obrazku
#if len(sys.argv) < 3:
#    print("Usage: python3 run.py <image_filename>")
#    sys.exit(1)
data_list=[]

if sys.argv[2] == "-v":
    for i in range(3,len(sys.argv)):
        image_filename = sys.argv[i]
        image = skimage.io.imread(image_filename,as_gray=True)
        img_og = skimage.io.imread(image_filename)
        img = np.array(image)



        if img_og.shape[0] > img_og.shape[1]:
            img_og = skimage.transform.rotate(img_og, 90, resize=True,)

        #Koontroloa jestli je obrazek natocen spravne
        if img.shape[0] > img.shape[1]:
            img = skimage.transform.rotate(img, 90, resize=True,)

        data_list.append(main(img,img_og,image_filename,True))
else:
    for i in range(2,len(sys.argv)):
        image_filename = sys.argv[i]
        image = skimage.io.imread(image_filename,as_gray=True)
        img_og = skimage.io.imread(image_filename)
        img = np.array(image)



        if img_og.shape[0] > img_og.shape[1]:
            img_og = skimage.transform.rotate(img_og, 90, resize=True,)

        #Koontroloa jestli je obrazek natocen spravne
        if img.shape[0] > img.shape[1]:
            img = skimage.transform.rotate(img, 90, resize=True,)

        data_list.append(main(img,img_og,image_filename,False))


output_file = sys.argv[1]
with open(output_file,'w') as fw:
   json.dump(data_list, fw)
print(f"Vysledna data zapsana do souboru {output_file}.")

