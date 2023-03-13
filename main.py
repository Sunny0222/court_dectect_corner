import math
import numpy as np
import cv2
#conda install -c conda-forge opencv
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter

#True:pic False:影像
ISPIC = True 

#影像大小 
#WIDTH self.bgr.shape[1]
#HEIGHT self.bgr.shape[0]
ZOOM_OUT = 2 #長寬縮小倍率 原始:3
# RANGE= range(9,41)#圖片index
# RANGE= [6,18]
RANGE= [29]

#前處理
SL=128  #白:255 -> 128:偏白 230:黃線排除
t1=6 #判斷前後左右多少pixel後開始檢查 初始:6
t2=20 #判斷是否為線段，可接受檢測到多少寬度 初始:10
sd=30 #判斷相差不超過sd的值，去雜質 初始20

#霍夫轉換
H_THRESHOLD = 100 #最短多少才能一條線 ini:500/6
H_MINLENGTH = 100 #最短多少才能一條線 ini:500/6
H_MAXGAP = 5 #允許線段的最大間隙 ini:5

#打包
PACKAGE_DIST=3  #打包線段時，取距離多少內，參考論文:5
PACKAGE_ANGLE=0.075  #打包線段時，取角度多少內，參考論文:0.75 實測:0.075

#畫出十字
CHOOS_LINE = 0 #0:畫出完整線段 1:劃出十字
CHECK_CROSS = 25 #為了檢測十字，線條內縮大小(pix)

#畫出交點
CIRCLE_SIZE=5 #畫交點半徑大小
CIRCLE_ELAST = 5 #交點抓取，有彈性:7 沒彈性:0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3),(1,1))

#計時
class CourtDetector(object):
    def __init__(self):
        self.yellow = 0,255,255 #把線單純畫出來顯示用  
        
    #讀取圖片
    def read_image(self, path):
        self.bgr = cv2.imread(path,cv2.IMREAD_COLOR)#3通道
        # self.bgr = cv2.imread(path.format("SBCC", 1))#3通道
        
        if self.bgr is None:
            print("圖片路徑無法讀取")
        else:
            print(f"圖片路徑:{path}")
    
    #讀取影像
    def getVideo(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Can't open camera !")
            exit()

        while(True):
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame")
                exit()

            self.bgr = frame 
            self.preProcess()
            self.detectLines()
            self.showCourt()

            keyName = cv2.waitKey(1)
            if keyName != -1:
                break
        cv2.destroyAllWindows()
        cap.release()

    #預處理
    def preProcess(self):
        self.bgr = cv2.resize(self.bgr, (int(self.bgr.shape[1]/ZOOM_OUT), int(self.bgr.shape[0]/ZOOM_OUT))) #事後再調整影像大小(避免運算失真)

        ycrcb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2YCrCb) #轉換成YCrCb顏色空間法(看明亮度)

        self.whiteMask = np.zeros((self.bgr.shape[0], self.bgr.shape[1],1), dtype = "uint8") #用來辨識白線 #單通道=灰度圖


        # RGB方法
        image = cv2.cvtColor(self.bgr,cv2.COLOR_BGR2HLS)
        lower = np.uint8([0, 200, 0])
        upper = np.uint8([255, 255, 255])
        self.whiteMask = cv2.inRange(image, lower, upper)
        
        # #ycrcb，取顏色偏白且相近的
        # for x in range(0, self.bgr.shape[1]):#左右內縮t3之後開始抓圖
        #     for y in range(0, self.bgr.shape[0]):#上下內縮t3之後開始抓圖
        #         isWhite = False
        #         here = ycrcb[y,x,0]; #Vec3b: 可描述RGB 3通道float類型的矩陣(BGR)

        #         if(here>SL): #白:255，所以設定>128則偏白

        #             for t in range(t1, t2+1): #看"前後左右"6~10，是否有色差不超過20的值: 去雜質點點
        #                 if x+t<ycrcb.shape[1]:
        #                     isWhite = isWhite or int(here)-int(ycrcb[y,x-t,0])>sd and int(here)-int(ycrcb[y,x+t,0])>sd
        #                 else:
        #                     isWhite = isWhite or int(here)-int(ycrcb[y,x-t,0])>sd
                        
        #                 if y+t<ycrcb.shape[0]:
        #                     isWhite = isWhite or int(here)-int(ycrcb[y-t,x,0])>sd and int(here)-int(ycrcb[y+t,x,0])>sd
        #                 else:
        #                     isWhite = isWhite or int(here)-int(ycrcb[y-t,x,0])>sd

        #         if(isWhite):
        #             self.whiteMask[y,x] = 255 #將有標到的地方弄白

        # self.whiteMask = cv2.dilate(self.whiteMask, kernel); #影像膨脹(二值化的影像,卷積核心的大小)
        # showImage(self.whiteMask)

    #線段偵測
    def detectLines(self):
        self.lineImg = np.zeros((self.bgr.shape[0], self.bgr.shape[1],3), dtype=np.uint8) #新圖:偵測後的白線
        self.lineImg.fill(255)

        #輸出線段(x1,y1,x2,y2)=線段頭尾 #參數可調整先參考demo
        #要以單通道辨識模型
        #霍夫轉換 (masked_edges，rho，theta，threshold，minLineLength，maxLineGap)
        self.lines = cv2.HoughLinesP(self.whiteMask, 1, np.pi/180, H_THRESHOLD, minLineLength=H_MINLENGTH, maxLineGap=H_MAXGAP)
        
        #線段分組
        # self.group_line_list = groupLine(self.lines)
        self.group_line_list = groupLine(self.lines)

        #將每組線段合併成一條線
        self.line_list=[] #最終的所有白線 ([min_reg_x,min_reg_y],[max_reg_x,max_reg_y],[所有白線上point的index & x座標(排序用)])
        self.line_list_short=[] #較短的所有白線 ([min_reg_x,min_reg_y],[max_reg_x,max_reg_y])
        for lines_i in range(len(self.group_line_list)):
            self.line_list, self.line_list_short = combineLine(self.line_list, self.line_list_short, self.group_line_list[lines_i])

        self.Verti_line = [] #垂直線段
        self.hori_line = [] #水平線段

        #檢查用rgb顏色
        rgb=[
        # [255,0,0],[255,145,0],[255,238,0],
        [0,100,0],[0, 255, 238],[0, 191, 255],
        [0, 30, 255],[139, 0, 255],[255, 0, 217],
        [210, 105, 30],[0, 255, 0],[217, 77, 255],
        # [255,255,255]
        ]
        #紅、橙、黃(已刪)
        #深綠、青、藍
        #海軍藍、紫、芭比粉
        #巧克力色、亮綠、亮紫
        #白(已刪)

        #rgb轉成bgr
        for rgb_i in rgb:
            rgb_i.reverse()

        #顯示全部白線
        self.point_cross = [] #交叉的所有交點
        self.point_cross_not = [] #沒交叉的所有交點
        self.point = [] #所有交點
        col_i=-1
        all_point_ind = -1 #交點index
        # for l in range(len(self.lines)): #測試全部線段
        for l in range(len(self.line_list)): #第一條線
            #畫線
            if col_i<len(rgb)-1: #線的color
                col_i = col_i+1
            drawWhiteLine(self.lineImg,self.line_list[l], rgb[col_i])
            # drawWhiteLine(self.lineImg,[(self.lines[l][0][0],self.lines[l][0][1]),(self.lines[l][0][2],self.lines[l][0][3])], rgb[col_i]) #測試全部線段
            
            #畫交點
            for point_i in range(l+1,len(self.line_list)): #第二條線
                if LineString([[self.line_list[l][0][0],self.line_list[l][0][1]],[self.line_list[l][1][0],self.line_list[l][1][1]]]).distance(LineString([[self.line_list[point_i][0][0],self.line_list[point_i][0][1]],[self.line_list[point_i][1][0],self.line_list[point_i][1][1]]])) <= CIRCLE_ELAST:  #是否為交點
                    inter_point = intersection(self.line_list[l],self.line_list[point_i]) #交點座標 且在白線上也附加
                    
                    if(inter_point[2]):
                        all_point_ind = all_point_ind + 1
                        is_cross = LineString(self.line_list_short[l]).distance(LineString(self.line_list_short[point_i])) <= 0 #是否為十字交點
                        self.line_list[l][2].append([all_point_ind,inter_point[0]])
                        self.line_list[point_i][2].append([all_point_ind,inter_point[0]])
                        
                        #把交點畫在線段圖片上
                        if is_cross: #十字設定紅色
                            self.point.append([all_point_ind, inter_point[0], inter_point[1], l, point_i, True]) #儲存交點: index, xy座標, 兩條線段index, 是否是cross
                            self.point_cross.append(all_point_ind)

                            cv2.circle(self.lineImg, (int(inter_point[0]),int(inter_point[1])), CIRCLE_SIZE, (0,0,255), -1) #畫circle一定是int
                        
                        else: #其他為橘色
                            self.point.append([all_point_ind, inter_point[0], inter_point[1], l, point_i, False]) #儲存交點: index, xy座標, 兩條線段index, 是否是cross
                            self.point_cross_not.append(all_point_ind) #儲存交點index
                            

                            cv2.circle(self.lineImg, (int(inter_point[0]),int(inter_point[1])), CIRCLE_SIZE, (0,145,255), -1) #畫circle一定是int
                            # cv2.circle(self.lineImg, (int(inter_point[0]),int(inter_point[1])), CIRCLE_SIZE, (0,0,255), -1) #畫circle一定是int


        #重新排序self.line_list內交點，才能尋找最近的
        for l in range(len(self.line_list)):
            self.line_list[l][2] = sorted(self.line_list[l][2], key=itemgetter(1))

        #找到目標cross
        #尋找每個十字交點
        p_edge_list = [] #邊緣點index

        for cross_i in range(len(self.point_cross)):
            l1 = self.point[self.point_cross[cross_i]][3] #交叉線段index
            l2 = self.point[self.point_cross[cross_i]][4] #交叉線段index
            
            #找到中繼點的候選人
            p1_list = []
            p2_list = []

            #找到同一線段左右交點(append前list, line_list, 線段index, 交點index, 是否為cross)  output:[交點index]
            p1_list = find_next_point(p1_list, self.line_list, self.point, l1, self.point_cross[cross_i], False)
            p2_list = find_next_point(p2_list, self.line_list, self.point, l2, self.point_cross[cross_i], False)

            #找到頂點的候選人
            p3_list = []
            p4_list = []
            
            if len(p1_list) == 0 or len(p2_list)==0:
                continue

            for p1 in p1_list:
                #找到左右交點的另一個線段
                if l1 == self.point[p1][3]:
                    l3 = self.point[p1][4]
                else:
                    l3 = self.point[p1][3]
                
                #找到同一條線的cross
                p3_list = find_next_point(p3_list,self.line_list, self.point, l3, p1, False)

            for p2 in p2_list:
                #找到左右交點的另一個線段
                if l2 == self.point[p2][3]:
                    l4 = self.point[p2][4]
                else:
                    l4 = self.point[p2][3]
                
                #找到同一條線的cross
                p4_list = find_next_point(p4_list,self.line_list, self.point, l4, p2, False)  

            #最終找到邊緣點
            p_edge_list = list(set(p3_list).intersection(set(p4_list)).union(set(p_edge_list)))
            
            # #原先的尋找交點 #舊版
            # #找到兩側交點，取另一邊線段
            # for p in self.point_cross_not:
            #     if p[2] == self.point_cross[cross_i][2]:
            #         if self.line_list[p[2]][2]
            #         p1.append(p[3])

            #     elif p[3] == self.point_cross[cross_i][2]:
            #         p1.append(p[2])

            #     elif p[2] == self.point_cross[cross_i][3]:
            #         p2.append(p[3])

            #     elif p[3] == self.point_cross[cross_i][3]:
            #         p2.append(p[2])
            
            # #找到邊緣點
            # if len(p1) !=0 and len(p2) != 0:
            #     find_edge(p_edge, cross_i, self.point_cross_not, p1, p2) #尋找邊緣，加到p_edge

        #計算向量(如何移動 or 沒有找到)
        if len(p_edge_list) != 0:
            edge_point = []
            #取y軸最大的邊緣點(最近)
            for p_edge_i in p_edge_list:

                edge_point.append(self.point[p_edge_i]) #抓取邊緣點

                # edge_ind = self.point[p_edge_i]
                # cv2.circle(self.lineImg, (int(edge_ind[1]),int(edge_ind[2])), CIRCLE_SIZE, (0,238,255), -1) #畫circle一定是int

            edge_ind = max(edge_point, key = lambda x : x[2])  
            #先設定邊緣點畫黃色
            cv2.circle(self.lineImg, (int(edge_ind[1]),int(edge_ind[2])), CIRCLE_SIZE, (0,238,255), -1) #畫circle一定是int
                        
    #總圖片顯示
    def showCourt(self):
        result = copyTo(self.bgr, self.lineImg) #將原圖 + 線段 + 交點
        showImage(result)
        # showImage(self.lineImg)

# #共用圖片顯示 #舊版
# def showImage(pic):
#     cv2.imshow("result", pic)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#共用圖片顯示 #舊版
def showImage(pic):
    cv2.imshow("result", pic)
    if(ISPIC):
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#共用畫線(output, 線段x1/y1/x2/y2 ,BGR顏色(預設白色))
def drawWhiteLine(mat,l,color=(255,255,255)):
    cv2.line(mat ,(int(l[0][0]),int(l[0][1])),(int(l[1][0]),int(l[1][1])) ,color ,1 ,cv2.LINE_AA)#在空白處繪製線條

# #共用線段分組(全部線段) #舊版
# def groupLine(lines):
#     line_list = [] #不知道怎麼處理三維ndarray所以用list
    
#     #把相似線段分群
#     for l1 in lines:
#         new_line = True #判斷是否已被分類

#         #群組包內是否為空
#         if len(line_list) == 0  and new_line:
#             line_list.append(l1)
#             new_line = False
#             continue

#         #瀏覽每個群組，看有沒有可以插入的(以index瀏覽)
#         l1_lineStr=LineString([(l1[0][0], l1[0][1]), (l1[0][2],l1[0][3])]) #用來計算下面distance
#         for l2_i in range(np.shape(line_list)[0]):
#             if new_line == False:
#                 break

#             for l2_i_j in line_list[l2_i]:#檢查群組內全部線斷是否有符合
#                 l2_lineStr=LineString([(l2_i_j[0], l2_i_j[1]), (l2_i_j[2],l2_i_j[3])])#用來計算下面distance
                
#                 #判斷是否包成一包
#                 # if l1_lineStr.distance(l2_lineStr)<5 and angle(l1[0], l2_i_j)<0.75:#參考論文的，但以我而言不行
#                 if l1_lineStr.distance(l2_lineStr)<PACKAGE_DIST and angle(l1[0], l2_i_j)<PACKAGE_ANGLE:
                    
                    
#                     #插入群組
#                     new_l2 = np.append(line_list[l2_i],l1,axis=0)
#                     line_list[l2_i] = new_l2
#                     new_line = False
#                     break

#         #以上皆非，新增一個新群組
#         if(new_line):
#             line_list.append(l1)
    
#     return line_list



#共用線段分組(全部線段)
def groupLine(lines):
    line_list = [] #不知道怎麼處理三維ndarray所以用list
    
    #排除沒抓到線的情況
    if lines is None:
        return line_list

    #把相似線段分群
    for l1 in lines:
        g_idx = [] #每個線段相似的群組(可能不只一個)

        #群組包內是否為空
        if len(line_list) == 0:
            line_list.append(l1)
            continue

        #瀏覽每個群組，看有沒有可以插入的(以index瀏覽)
        l1_lineStr=LineString([(l1[0][0], l1[0][1]), (l1[0][2],l1[0][3])]) #用來計算下面distance
        for l2_i in range(len(line_list)):

            for l2_i_j in line_list[l2_i]:#檢查群組內全部線斷是否有符合
                l2_lineStr=LineString([(l2_i_j[0], l2_i_j[1]), (l2_i_j[2],l2_i_j[3])])#用來計算下面distance
                
                #判斷是否包成一包
                # if l1_lineStr.distance(l2_lineStr)<5 and angle(l1[0], l2_i_j)<0.75:#參考論文的，但以我而言不行
                if l1_lineStr.distance(l2_lineStr)<PACKAGE_DIST and angle(l1[0], l2_i_j)<PACKAGE_ANGLE:
                    #判斷那些群組有符合
                    g_idx.append(l2_i)
                    break
            
        if len(g_idx) !=0:

            g_idx.sort()
            new_g_list = []
            
            #將兩個群組合併
            for g_idx_i in range(len(g_idx)):
                if len(new_g_list)==0:
                    new_g_list = line_list[g_idx[g_idx_i]-g_idx_i] #因為被移除所以減index
                else:
                    new_g_list = np.append(new_g_list,line_list[g_idx[g_idx_i]-g_idx_i],axis=0)

                del line_list[g_idx[g_idx_i]-g_idx_i]

            new_g_list = np.append(new_g_list, l1, axis=0)
            
            line_list.append(new_g_list)

        #以上皆非，新增一個新群組
        else:
            line_list.append(l1)

    return line_list

#共用群組線段合併(line_list, line_list_short, ndarray)
#output: [(x1,y1), (x2,y2), (x1,y1)-n pix, (x2,y2)-n pix]
def combineLine(line_list, line_list_short, lines):

    #抓出點集合
    points = []
    for line in lines:
        x1, y1, x2, y2 = line
        points.append([x1,y1])
        points.append([x2,y2])
    points = np.array(points)

    #輸出線段
    # result = []
    
    [vx,vy,x,y] = cv2.fitLine(points,cv2.DIST_L2,0,0.01,0.01) #直線擬合:找到最小dist的回歸線for all 點
    
    #畫出回歸線(兩邊盡頭是整張圖)
    # #由比較右邊的點start
    # lefty = int((-x*vy/vx) + y)
    # righty = int(((WIDTH-x)*vy/vx)+y)
    # line_mask = np.zeros([HEIGHT,WIDTH], dtype=np.uint8)
    # cv2.line(line_mask,(WIDTH-1,righty),(0,lefty),[255,255,255],2)  #不知道為甚麼要-1
    # showImage(line_mask)
    # result.append(np.array([WIDTH,righty,0,lefty], dtype=np.int32)) #取得最佳全圖回歸線

    #畫出回歸線(盡頭是真實線段) #xy軸是鏡像所以可以照著一般邏輯反轉max,min
    if vx == 0: #垂直
        reg_all_y = []
        for reg_point in points: 
            reg_all_y.append(reg_point[1])

        min_reg_x = x
        max_reg_x = x
        min_reg_y = min(reg_all_y)
        max_reg_y = max(reg_all_y)
    
    elif vy == 0: #平行
        reg_all_x = []
        for reg_point in points: 
            reg_all_x.append(reg_point[0])

        min_reg_x = min(reg_all_x)
        max_reg_x = max(reg_all_x)
        min_reg_y = y
        max_reg_y = y

    else: #其他
        b = int((-x*vy/vx) + y) #找到 y=ax+b的b
        b = (-x*vy/vx) + y #找到 y=ax+b的b

        reg_all_y = []#找到最大與最小回歸線上的y
        for reg_point in points: 
            reg_all_y.append(vy/vx * reg_point[0] + b ) #計算每個x相對的y

        min_reg_y = min(reg_all_y)
        max_reg_y = max(reg_all_y)

        min_reg_x = (min_reg_y - b) * vx/vy
        max_reg_x = (max_reg_y - b) * vx/vy

    #輸出完整線條
    line_list.append([[min_reg_x,min_reg_y],[max_reg_x,max_reg_y],[]])
    
    #內縮畫十字，線段減n pix
    theta = math.atan2(vy,vx) #用向量會在180度之間
    cos = math.cos(theta)
    sin = math.sin(theta)
    if min_reg_x < max_reg_x:
        min_reg_x = min_reg_x + cos*CHECK_CROSS
        min_reg_y = min_reg_y + sin*CHECK_CROSS
        max_reg_x = max_reg_x - cos*CHECK_CROSS
        max_reg_y = max_reg_y - sin*CHECK_CROSS
    else:
        min_reg_x = min_reg_x - cos*CHECK_CROSS
        min_reg_y = min_reg_y - sin*CHECK_CROSS
        max_reg_x = max_reg_x + cos*CHECK_CROSS
        max_reg_y = max_reg_y + sin*CHECK_CROSS

    #輸出減掉CHECK_CROSS的線條
    line_list_short.append([[min_reg_x,min_reg_y],[max_reg_x,max_reg_y]])

    return line_list, line_list_short

#共用找出垂直線(線段x1/y1/x2/y2)
def isVertical(l):
    return (l[0]-l[2])*(l[0]-l[2]) < (l[1]-l[3])*(l[1]-l[3])

#共用找交點(線1，線2) #向量相交即可，不用確切交點
def intersection(l1, l2):
    # 第1條線: l1[0], l1[1], l1[2], l1[3] = (起始點x軸, 起始點y軸, 結束點x軸, 結束點y軸)
    # 第2條線: l2[0], l2[1], l2[2], l2[3] = (起始點x軸, 起始點y軸, 結束點x軸, 結束點y軸)
    a1 = l1[1][1]-l1[0][1]
    b1 = l1[0][0]-l1[1][0]
    c1 = a1*l1[0][0] + b1*l1[0][1]
    
    a2 = l2[1][1]-l2[0][1]
    b2 = l2[0][0]-l2[1][0]
    c2 = a2*l2[0][0] + b2*l2[0][1]

    # a1 = l1[3]-l1[1]
    # b1 = l1[0]-l1[2]
    # c1 = a1*l1[0] + b1*l1[1]
    
    # a2 = l2[3]-l2[1]
    # b2 = l2[0]-l2[2]
    # c2 = a2*l2[0] + b2*l2[1]

    det = a1*b2 - a2*b1 #行列式

    #公式:克萊姆法則 可參考 https://badvertex.com/2013/06/24/determining-if-two-lines-intersect.html
    if(det != 0): #行列式為0: 平行
        x = (b2*c1 - b1*c2)/det
        y = (a1*c2 - a2*c1)/det

        return (x,y,True)
    else: 
        return (0,0,False)

#共用計算最小的角度(線1，線2)
def angle(l1, l2):
    # 第1條線: l1[0], l1[1], l1[2], l1[3] = (起始點x軸, 起始點y軸, 結束點x軸, 結束點y軸)
    # 第2條線: l2[0], l2[1], l2[2], l2[3] = (起始點x軸, 起始點y軸, 結束點x軸, 結束點y軸)
    # Get nicer vector form
    v1 = [(l1[0]-l1[2]), (l1[1]-l1[3])]
    v2 = [(l2[0]-l2[2]), (l2[1]-l2[3])]
    
    #作法二
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)

#共用把mask疊到原圖上面 (原圖,附加非黑圖案&形狀)
def copyTo(bgr, ori_pic):

    img_gray = cv2.cvtColor(ori_pic, cv2.COLOR_BGR2GRAY) # 產生一張灰階的圖片作為遮罩使用
    ret, mask1  = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)  # 使用二值化的方法，產生黑白遮罩圖片
    mask_ori_pic = cv2.bitwise_and(ori_pic, ori_pic, mask = mask1 )  # 產生黑圖+遮罩區域

    ret, mask2  = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)      # 使用二值化的方法，產生黑白遮罩圖片
    bgr = cv2.bitwise_and(bgr, bgr, mask = mask2 )      # 產生底圖-遮罩區域
    
    result = cv2.add(bgr, mask_ori_pic)                       # 將底圖和遮罩合併

    return result

#尋找邊緣 input: 總邊緣, 交叉邊緣index, 非交叉邊緣list, 相交點1, 相交點2
def find_edge(p_edge, cross_i, point_cross_not,p1 ,p2):
    for p_edge_i in range(len(point_cross_not)):
        for p1_i in p1:
            if p1_i == point_cross_not[p_edge_i][2]:
                for p2_i in p2:
                    if p2_i == point_cross_not[p_edge_i][3]:
                        p_edge.append((cross_i,p_edge_i))
                        return p_edge
            elif p1_i == point_cross_not[p_edge_i][3]:
                for p2_i in p2:
                    if p2_i == point_cross_not[p_edge_i][2]:
                        p_edge.append((cross_i,p_edge_i))
                        return p_edge

#找到同一線段左右交點(result, line_list, 線段index, 交點index, 是否為Cross) output:[交點index]
def find_next_point(result, line_list, point_list, l1, cross_i, is_cross):
    ind_p1 = -1 #交點index

    for p1 in line_list[l1][2]:
        ind_p1 = ind_p1+1
        
        #get p1為線段上的點
        if p1[0] == cross_i : 
            if ind_p1+1 == len(line_list[l1][2]): #index尾端
                new_p1 = ind_p1 - 1
                
                if point_list[line_list[l1][2][new_p1][0]][5] == is_cross: #要抓取Cross or other
                    result.append(line_list[l1][2][new_p1][0])

                
            elif ind_p1 == 0: #index頭
                new_p1 = 1
                
                if point_list[line_list[l1][2][new_p1][0]][5] == is_cross: #要抓取Cross or other
                    result.append(line_list[l1][2][new_p1][0])
            
            else:
                new_p1_1 = ind_p1 - 1
                new_p1_2 = ind_p1 + 1

                if point_list[line_list[l1][2][new_p1_1][0]][5] == is_cross: #要抓取Cross or other
                    result.append(line_list[l1][2][new_p1_1][0])

                if point_list[line_list[l1][2][new_p1_2][0]][5] == is_cross:
                    result.append(line_list[l1][2][new_p1_2][0])
    return result

#讀取照片
def processImage(path):
    img = CourtDetector()
    img.read_image(path)
    img.preProcess()
    img.detectLines()
    img.showCourt()

#讀取影像
def processVideo():
    img = CourtDetector()
    img.getVideo()

    # # 以下皆包在getVideo內
    # img.preProcess()
    # img.detectLines()
    # img.showCourt()

if __name__ == "__main__":

    if ISPIC:
        #讀取照片
        for i in RANGE:
            processImage(f"D:\\table\\SiliconAwards\\badminton_pic\\school_pic\\test{i}.jpg")

    else: 
        #讀取影像
        processVideo()
    
