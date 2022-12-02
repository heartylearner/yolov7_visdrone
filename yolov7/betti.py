
import time
#from pycuda import driver
import argparse
import os
import numpy as np
from copy import deepcopy
device_idx=0
devices=list(map(int,os.environ["CUDA_VISIBLE_DEVICES"].split(','))) # specify which GPU(s) to be used
devices_count=len(devices)
import cv2
#import pycuda
#import pycuda.driver as cuda
import time
#cuda.init()
thres=0.5
split_size=1024
from torch.multiprocessing import Process
import multiprocessing as mp
from multiprocessing import shared_memory
#import skimage.measure
images=[
    [
        [[1,0,0],[1,0,0],[1,0,0]],
        [[1,0,0],[0,1,0],[1,0,0]],
        [[1,0,0],[1,0,0],[1,0,0]],
    ],
    [
        [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]],
        [[1,0,0],[1,0,0],[0,1,0],[1,0,0],[1,0,0]],
        [[1,0,0],[0,1,0],[0,1,0],[0,1,0],[1,0,0]],
        [[1,0,0],[1,0,0],[0,1,0],[1,0,0],[1,0,0]],
        [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]]
    ]
]
# class gpuThread(threading.Thread):
#     def __init__(self, gpuid,f,id,P,S,u0,v0,u1,v1,U0,V0,W,H,BC):
#         threading.Thread.__init__(self)
#         print("gpuid=",gpuid)
#         self.f=f
#         self.id=id
#         self.P=P
#         self.S=S
#         self.u0=u0
#         self.u1=u1
#         self.v0=v0
#         self.v1=v1
#         self.U0,self.V0=U0,V0
#         self.W,self.H=W,H
#         self.ctx  = driver.Device(gpuid).make_context()
#         self.device = self.ctx.get_device()
#         self.BC=BC
#     def run(self):
#         self.t=time.time()
#         self.f(self.id,self.P,self.S,self.u0,self.v0,self.u1,self.v1,self.U0,self.V0,self.W,self.H,self.BC)
        
        # Profit!

    # def join(self):
    #     self.ctx.detach()
    #     threading.Thread.join(self)
    #     print(f"thread u0={self.u0} v0={self.v0} u1={self.u1} v1={self.v1} run {time.time()-self.t} s")
def device_assign():
    global device_idx
    device_idx=(device_idx+1)%devices_count
    return device_idx
def dist(p1,p2):
    d=0
    for i in range(len(p1)):
        d+=(int(p1[i])-int(p2[i]))**2
    return d**0.5

def find(P,c):
    if(int(P[c])==-1):
        return c
    P[c]=find(P,int(P[c]))
    return int(P[c])

def union(P,S,u0,v0,u1,v1,U0,V0,W,H,BC):
    c1,c2=u0-U0+(v0-V0)*W,u1-U0+(v1-V0)*W
    #print("edge",c1,c2)
    h1,h2=find(P,c1),find(P,c2)
    if(h1==h2):
        BC[0]+=1
    else:
        if(S[h1]>S[h2]):
            P[h2]=h1
            S[h1]+=S[h2]
        else:
            P[h1]=h2
            S[h2]+=S[h1]

def mergeu(id,P,S,u0,v0,u1,v1,U0,V0,W,H,BC):
    #print("mergeu",u0,v0,u1,v1,BC)
    d0111=dist(images[id][v0][u0],images[id][v0][u1])
    for v in range(v0,v1):
        d0010=d0111
        d0111=dist(images[id][v+1][u0],images[id][v+1][u1])
        d0110=dist(images[id][v+1][u0],images[id][v][u1])
        d0011=dist(images[id][v][u0],images[id][v+1][u1])
        # if(v==v0):
        #     print("first round step 0",P[u0-U0],P[u1-U0],P[u0-U0+W],P[u1-U0+W])
        if(d0010<=thres):union(P,S,u0,v,u1,v,U0,V0,W,H,BC)
        
        # if(v==v0):
        #     print("first round step 1",u0,v0,u1,v1,BC[0])
        if(d0110<=thres):
            union(P,S,u0,v+1,u1,v,U0,V0,W,H,BC)
            # if(v==v0):
            #     print("first round step 2",u0,v0,u1,v1,BC[0])
            d0001=dist(images[id][v][u0],images[id][v+1][u0])
            d1011=dist(images[id][v][u1],images[id][v+1][u1])
            if(d0010<=thres and d0001<=thres):BC[0]=int(BC[0])-1
            if(d0111<=thres and d1011<=thres):BC[0]=int(BC[0])-1
            # if(v==v0):
            #     print("first round step 3",u0,v0,u1,v1,BC[0])

        elif(d0011<=thres):
            union(P,S,u0,v,u1,v+1,U0,V0,W,H,BC)
            d0001=dist(images[id][v][u0],images[id][v+1][u0])
            d1011=dist(images[id][v][u1],images[id][v+1][u1])
            if(d0010<=thres and d1011<=thres):BC[0]=int(BC[0])-1
            if(d0001<=thres and d0111<=thres):BC[0]=int(BC[0])-1
        # if(v==v0):
        #     print("first round",u0,v0,u1,v1,BC[0])
    if(d0111<=thres):union(P,S,u0,v1,u1,v1,U0,V0,W,H,BC)
    #print("after mergeu",u0,v0,u1,v1,BC)
def mergev(id,P,S,u0,v0,u1,v1,U0,V0,W,H,BC):
    #print("mergev",BC)
    d1011=dist(images[id][v0][u0],images[id][v1][u0])
    for u in range(u0,u1):
        d0001=d1011
        d1011=dist(images[id][v0][u+1],images[id][v1][u+1])
        d0110=dist(images[id][v1][u],images[id][v0][u+1])
        d0011=dist(images[id][v0][u],images[id][v1][u+1])
        if(d0001<=thres):union(P,S,u,v0,u,v1,U0,V0,W,H,BC)
        # if(u==u0):
        #     print("first round step 1",u0,v0,u1,v1,BC[0])
        if(d0110<=thres):
            union(P,S,u,v1,u+1,v0,U0,V0,W,H,BC)
            d0010=dist(images[id][v0][u],images[id][v0][u+1])
            d0111=dist(images[id][v1][u],images[id][v1][u+1])
            if(d0010<=thres and d0001<=thres):BC[0]=int(BC[0])-1
            if(d0111<=thres and d1011<=thres):BC[0]=int(BC[0])-1

        elif(d0011<=thres):
            union(P,S,u,v0,u+1,v1,U0,V0,W,H,BC)
            # if(u==u0):
            #     print("first round step 2",u0,v0,u1,v1,BC[0])
            d0010=dist(images[id][v0][u],images[id][v0][u+1])
            d0111=dist(images[id][v1][u],images[id][v1][u+1])
            if(d0010<=thres and d1011<=thres):BC[0]=int(BC[0])-1
            if(d0001<=thres and d0111<=thres):BC[0]=int(BC[0])-1
            # if(u==u0):
            #     print("first round step 3",d0001,d0111,images[id][v0][u],images[id][v0][u+1],images[id][v1][u],images[id][v1][u+1],u0,v0,u1,v1,BC[0])
        # if(u==u0):
        #     print("first round",u0,v0,u1,v1,BC[0])
    if(d1011<=thres):union(P,S,u1,v0,u1,v1,U0,V0,W,H,BC)
    #print("after mergev",BC)
def check(P,S,u0,v0,u1,v1,U0,V0,W,H):
    C=0
    for u in range(u0,u1+1):
        for v in range(v0,v1+1):
            c=u-U0+(v-V0)*W
            if(P[c]==-1):
                C+=1
            if(C>1):
                return c
    return -1
def cal(id,P,S,u0,v0,u1,v1,U0,V0,W,H,BC,num_workers=1):
    #print(f"cal id={id} u0={u0} v0={v0} u1={u1} v1={v1} BC={BC[0]}")
    if(u1-u0>=split_size and num_workers>1):
        um=(u1+u0)//2
        #BC1=deepcopy(BC)
        # sm1=shared_memory.SharedMemory(create=True, size=2)
        # sm2=shared_memory.SharedMemory(create=True, size=2)
        # BC1=sm1.buf
        # BC2=sm2.buf
        #BC1[0]=-1
        # BC1[0]=0
        # BC2[0]=0
        #BC1=mp.Array('i', [0])
        BC2=mp.Array('i', [0])
        #t0=gpuThread(device_assign(),cal,id,P,S,u0,v0,um,v1,U0,V0,W,H,BC)
        #t1=gpuThread(device_assign(),cal,id,P,S,um+1,v0,u1,v1,U0,V0,W,H,BC1)
        #BCshared_memory.SharedMemory(create=True, size=1)
        t0=Process(target=cal,args=(id,P,S,u0,v0,um,v1,U0,V0,W,H,BC,num_workers//2))
        t1=Process(target=cal,args=(id,P,S,um+1,v0,u1,v1,U0,V0,W,H,BC2,num_workers//2))
        t0.start()
        t1.start()
        t0.join()
        t1.join()
        BC[0]+=BC2[0]
        #BC1.close()
        #BC2.close()
        # sm1.close()
        # sm2.close()
        # sm1.unlink()
        # sm2.unlink()
        t=time.time()
        #print(f"before mergeu u0={um} v0={v0} u1={um+1} v1={v1} BC={BC[0]}")
        
        mergeu(id,P,S,um,v0,um+1,v1,U0,V0,W,H,BC)
        #print(f"after mergeu u0={um} v0={v0} u1={um+1} v1={v1} BC={BC[0]} {time.time()-t}s")
    elif(v1-v0>=split_size and num_workers>1):
        vm=(v1+v0)//2
        #BC1=deepcopy(BC)
        #sm1=shared_memory.SharedMemory(create=True, size=2)
        #sm2=shared_memory.SharedMemory(create=True, size=2)
        # BC1=sm1.buf
        # BC2=sm2.buf
        # BC1[0]=0
        # BC2[0]=0
        #BC1=mp.Array('i', [0])
        BC2=mp.Array('i', [0])
        #t0=gpuThread(device_assign(),cal,id,P,S,u0,v0,u1,vm,U0,V0,W,H,BC)
        #t1=gpuThread(device_assign(),cal,id,P,S,u0,vm+1,u1,v1,U0,V0,W,H,BC1)
        t0=Process(target=cal,args=(id,P,S,u0,v0,u1,vm,U0,V0,W,H,BC,num_workers//2))
        t1=Process(target=cal,args=(id,P,S,u0,vm+1,u1,v1,U0,V0,W,H,BC2,num_workers//2))
        t0.start()
        t1.start()
        t0.join()
        t1.join()
        BC[0]+=BC2[0]
        #print(f"before mergev u0={u0} v0={vm} u1={u1} v1={vm+1} BC={BC[0]}")
        #BC1.close()
        #BC2.close()
        # sm1.close()
        # sm2.close()
        # sm1.unlink()
        # sm2.unlink()
        t=time.time()
        #BC2.unlink()
        mergev(id,P,S,u0,vm,u1,vm+1,U0,V0,W,H,BC)
        #print(f"after mergev u0={u0} v0={vm} u1={u1} v1={vm+1} BC={BC[0]} {time.time()-t}s")
    else:
        for u in range(u0,u1):
            for v in range(v0,v1):
                d0010=dist(images[id][v][u],images[id][v][u+1])
                if(d0010<=thres):union(P,S,u,v,u+1,v,U0,V0,W,H,BC)
                d0001=dist(images[id][v][u],images[id][v+1][u])
                if(d0001<=thres):union(P,S,u,v,u,v+1,U0,V0,W,H,BC)
                d0110=dist(images[id][v+1][u],images[id][v][u+1])
                d0011=dist(images[id][v][u],images[id][v+1][u+1])
                d0111=-1
                d1011=-1
                if(d0110<=thres):
                    union(P,S,u+1,v,u,v+1,U0,V0,W,H,BC)
                    d0111=dist(images[id][v+1][u],images[id][v+1][u+1])
                    d1011=dist(images[id][v][u+1],images[id][v+1][u+1])
                    if(d0010<=thres and d0001<thres):BC[0]-=1
                    if(d0111<=thres and d1011<=thres):BC[0]-=1
                elif(d0011<=thres):
                    union(P,S,u,v,u+1,v+1,U0,V0,W,H,BC)
                    d0111=dist(images[id][v+1][u],images[id][v+1][u+1])
                    d1011=dist(images[id][v][u+1],images[id][v+1][u+1])
                    if(d0010<=thres and d1011<thres):BC[0]-=1
                    if(d0111<=thres and d0001<=thres):BC[0]-=1
                # if((u0,v0,u1,v1)==(0,0,8,8)):
                #     print(u,v,d0010,d0001,d0110,d0011,d0111,d1011,BC)
            if(d0111<=thres):
                union(P,S,u,v1,u+1,v1,U0,V0,W,H,BC)
                # if((u0,v0,u1,v1)==(0,0,8,8)):
                #     print(u,v1,d0010,d0001,d0110,d0011,d0111,d1011,BC)
        for v in range(v0,v1):
            d0001=dist(images[id][v][u1],images[id][v+1][u1])
            if(d0001<=thres):union(P,S,u1,v,u1,v+1,U0,V0,W,H,BC)
            # if((u0,v0,u1,v1)==(0,0,8,8)):
            #     print(u1,v,BC)
        # check_id=check(P,S,u0,v0,u1,v1,U0,V0,W,H)
        # if(check_id!=-1):
        #     print(u0,v0,u1,v1,"fail at",check_id%W,check_id//W)

    #print(f"cal id={id} u0={u0} v0={v0} u1={u1} v1={v1} BC={BC[0]}")
        

def betti(id,pool,pool_idx,u0,v0,u1,v1,num_workers=1):
    #BC=shared_memory.SharedMemory(create=True, size=1)
    BC=mp.Array('i',[0],lock=False)
    P=-1*np.ones(((v1-v0+1)*(u1-u0+1),))
    P=mp.Array('f',P.tolist(),lock=False)
    S=np.ones(((v1-v0+1)*(u1-u0+1),))
    S=mp.Array('f',S.tolist(),lock=False)
    cal(id,P,S,u0,v0,u1,v1,u0,v0,u1-u0+1,v1-v0+1,BC,num_workers)
    #print(P)
    #print(S)
    #print("betti return ",BC[0])
    pool[pool_idx]=BC[0]
def betti_pool(id,w,h,num_workers):
    #print("in betti",images[id][0][0])
    Nu=len(images[id][0])//w
    Nh=len(images[id])//h
    pool=mp.Array('i',[0 for i in range(Nu*Nh)],lock=False)
    processes=[]
    c=0
    for i in range(Nu*Nh):
        u0=w*(i%Nu)
        v0=h*(i//Nu)
        #processes.append(Process(target=betti,args=(id,pool,i,u0,v0,min(u0+w,len(images[id][0])-1),min(v0+h,len(images[id])-1),1)))
        processes.append(Process(target=betti,args=(id,pool,i,u0,v0,min(u0+w-1,len(images[id][0])-1),min(v0+h-1,len(images[id])-1),(num_workers-Nu*Nh)//(Nu*Nh))))
        processes[-1].start()
        c+=1
        if(c==num_workers):
            for process in processes:
                process.join()
            processes=[]
            c=0
    pool=np.array(pool).reshape((Nh,Nu))
    return pool

def betti_conv(id,w,h,stride,num_workers):
    #print("in betti",images[id][0][0])
    Nu=((len(images[id][0])-w)//stride)+1
    Nh=((len(images[id])-h)//stride)+1
    pool=mp.Array('i',[0 for i in range(Nu*Nh)],lock=False)
    processes=[]
    c=0
    for i in range(Nu*Nh):
        u0=stride*(i%Nu)
        v0=stride*(i//Nu)
        #processes.append(Process(target=betti,args=(id,pool,i,u0,v0,min(u0+w,len(images[id][0])-1),min(v0+h,len(images[id])-1),1)))
        processes.append(Process(target=betti,args=(id,pool,i,u0,v0,min(u0+w-1,len(images[id][0])-1),min(v0+h-1,len(images[id])-1),(num_workers-Nu*Nh)//(Nu*Nh))))
        processes[-1].start()
        c+=1
        if(c==num_workers):
            for process in processes:
                process.join()
            processes=[]
            c=0
    pool=np.array(pool).reshape((Nh,Nu))
    return pool
def draw_box(img,u0,v0,w,h):
    for u in range(u0,min(u0+w,len(img[0]))):
        img[v0][u]=img[v0+1][u]=img[min(v0+h-1,len(img)-1)][u]=img[min(v0+h,len(img)-1)][u]=[255,0,0]
    for v in range(v0,min(v0+h,len(img))):
        img[v][u0]=img[v][u0+1]=img[v][min(u0+w-1,len(img[0])-1)]=img[v][min(u0+w,len(img[0])-1)]=[255,0,0]
def draw_boxes(img,pool,thres):
    W=len(img[0])//pool.shape[1]
    H=len(img)//pool.shape[0]
    for i in range(pool.shape[0]):
        for j in range(pool.shape[1]):
            if(int(pool[i,j])>thres):
                r=img[H*i:H*i+H,W*j:W*j+W,0]
                g=img[H*i:H*i+H,W*j:W*j+W,1]
                b=img[H*i:H*i+H,W*j:W*j+W,2]
                r=cv2.equalizeHist(r).reshape((H,W,1))
                g=cv2.equalizeHist(g).reshape((H,W,1))
                b=cv2.equalizeHist(b).reshape((H,W,1))
                img[H*i:H*i+H,W*j:W*j+W]=np.concatenate((r,g,b),axis=-1)
                #draw_box(img,W*j,H*i,W,H)
                
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--dist_thres', type=float, default=10, help='object confidence threshold')
    parser.add_argument('--target_size', type=int, default=80, help='object confidence threshold')
    parser.add_argument('--conv_stride', type=int, default=40, help='object confidence threshold')
    parser.add_argument('--box_thres', type=float, default=0, help='object confidence threshold')
    

    args = parser.parse_args()
    print(args)
    thres=args.dist_thres
    image=cv2.imread(args.source)
    img0=deepcopy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)


    image=cv2.medianBlur(image,15)
    image1=cv2.GaussianBlur(image,(5,5),0)
    image=(4*np.array(image)-3*np.array(image1))
    # cv2.imwrite("./g.jpg",image)
    
    #cv2.imwrite("./m_rg.jpg",image)
    #image=cv2.equalizeHist(image)
    #cv2.imwrite("./e.jpg",image)

    image=image.reshape((640,640,1))
    #thres=10
    images=[image]+images
    pool80=betti_pool(0,args.target_size,args.target_size,64)
    pool20=betti_pool(0,args.target_size//4,args.target_size//4,64)
    print("pool80\n",pool80)
    print("\npool20\n",pool20)
    #print(pool20[28:32,28:32])
    m, n = pool20.shape
    pool20_80 = np.transpose(pool20.reshape(m//4, 4, n//4, 4),axes=(0,2,1,3)).reshape(640//args.target_size,640//args.target_size,16)
    pool20_80=np.sum(pool20_80,axis=-1)
    print("\npool20_80\n",pool20_80)
    print("\npool80-pool20_80\n",pool80-pool20_80)
    #img_pool=img0.tolist()
    img_pool=deepcopy(img0)
    print(img_pool.shape)
    draw_boxes(img_pool,pool80-pool20_80,args.box_thres)
    cv2.imwrite(args.source[:-4]+f"_dist_{args.dist_thres}_betti_{args.box_thres}_pool.jpg",np.array(img_pool))
    #cv2.imwrite("./m_rg_bbox.jpg",np.array(img0))
    exit()
    conv80=betti_conv(0,args.target_size,args.target_size,args.conv_stride,64)
    print("\nconv80\n",conv80.shape,"\n",conv80)
    m,n=conv80.shape
    conv20=betti_conv(0,args.target_size//4,args.target_size//4,args.conv_stride//4,64)[:m*4,:n*4]
    print("\nconv20\n",conv20.shape,"\n",conv20)    
    m,n=conv20.shape
    conv20_80= np.transpose(conv20.reshape(m//4, 4, n//4, 4),axes=(0,2,1,3)).reshape((640-args.target_size)//args.conv_stride+1,(640-args.target_size)//args.conv_stride+1,16)
    conv20_80=np.sum(conv20_80,axis=-1)
    print("\nconv20_80\n",conv20_80.shape,"\n",conv20_80)
    #exit()
    print("\nconv80-conv20_80\n",conv80-conv20_80)
    img_conv=img0.tolist()
    draw_boxes(img_conv,conv80-conv20_80,args.box_thres)
    cv2.imwrite(args.source[:-4]+f"_dist_{args.dist_thres}_betti_{args.box_thres}_conv.jpg",np.array(img_conv))
    
#check_requirements(exclude=('pycocotools', 'thop'))

# test_img_size=640
# split_size=32
# img=[[[1] for i in range(test_img_size)]for i in range(test_img_size)]
# for i in range(test_img_size):
#     for j in range(test_img_size):
#         if(i%3==1 and j%3==1):
#             img[i][j]=[0]
# images=[img]+images
# t=time.time()
# id=0
# print(betti(id,0,0,len(images[id][0])-1,len(images[id])-1,128))
# betti_pool(0,64)
# print(betti(id,0,0,320,320,0))
# print(time.time()-t,"s")
# print(np.squeeze(np.array(img)))
