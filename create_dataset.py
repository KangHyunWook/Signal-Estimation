import os

class IMU:
    def __init__(self, config):
        
        self.config=config
        
        items=os.listdir(config.dataset_dir)
        items = sorted(items)
        
        pattern1_imu_path=os.path.join(config.dataset_dir,items[2])
        pattern1_GT_RR_path=os.path.join(config.dataset_dir,items[0])
        
        pattern2_imu_path=os.path.join(config.dataset_dir,items[3])
        pattern2_GT_RR_path=os.path.join(config.dataset_dir,items[1])
        
        
        self.imu_pattern1, self.GT_RRs_pattern1 = self.preprocess(pattern1_imu_path, pattern1_GT_RR_path)
        self.imu_pattern2, self.GT_RRs_pattern2 = self.preprocess(pattern2_imu_path, pattern2_GT_RR_path)
        
    def preprocess(self, IMU_PATH, GT_RR_PATH):
        
        imu_signal=[]
        accelX_list=[]
        accelY_list=[]
        accelZ_list=[]
        gyroX_list=[]
        gyroY_list=[]
        gyroZ_list=[]

        for line in open(IMU_PATH):
            line=line.strip()
            splits = line.split(',')    
            
            if splits[0]=='accelx':
                continue
            
            accelx = float(splits[0])
            accely = float(splits[1])
            accelz = float(splits[2])
            gyrox = float(splits[3])
            gyroy = float(splits[4])
            gyroz = float(splits[5])
            
            # imu_signal.append(np.array([splits[0],splits[1],splits[2], gyrox, gyroy, gyroz]))
            # imu_signal.append(accely)
            
            accelX_list.append(accelx)
            accelY_list.append(accely)
            accelZ_list.append(accelz)
            gyroX_list.append(gyrox)
            gyroY_list.append(gyroy)
            gyroZ_list.append(gyroz)


        thres=15000
        # print(len(accelX_list))
        # exit()
        self.accelX_list=accelX_list = accelX_list[:thres]
        self.accelY_list=accelY_list = accelY_list[:thres]
        self.accelZ_list=accelZ_list = accelZ_list[:thres]
        self.gyroX_list=gyroX_list = gyroX_list[:thres]
        self.gyroY_list=gyroY_list = gyroY_list[:thres]
        self.gyroZ_list=gyroZ_list = gyroZ_list[:thres]
        imu = (self.accelX_list, self.accelY_list, self.accelZ_list, 
                    self.gyroX_list, self.gyroY_list, self.gyroZ_list)
        
        
        #get GT RRs
        
        GT_RR_list=[]

        cnt=0
        for line in open(GT_RR_PATH):
            line = line.strip()
            splits = line.split(',')
            if splits[0]=='time':
                continue
            if cnt<30:
                GT_RR_list.append(float(splits[1]))
            cnt+=1
        
        return [imu,GT_RR_list]
                    
    def get_data(self):
        return [self.imu_pattern1, self.GT_RRs_pattern1, self.imu_pattern2, self.GT_RRs_pattern2]
    
    
class MIC:
    def __init__(self,config):
        DATA_PATH=config.dataset_dir

        time_list=[]
        sensor_list=[]
        GT_list=[] 

        for line in open(DATA_PATH):
            splits=line.split(',')
            time=splits[0]
            if 'Time' in time:
                continue
            if time=='':
                break
            time=float(time)
            sensor=float(splits[1])
            GT=float(splits[2])
            time_list.append(time)
            sensor_list.append(sensor)
            GT_list.append(GT)
            
                
        self.time_list=time_list
        self.sensor_list=sensor_list
        self.GT_list=GT_list

    def get_data(self):
        return (self.time_list, self.sensor_list, self.GT_list)