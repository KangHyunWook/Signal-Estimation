class IMU:
    def __init__(self, config):
        imu_signal=[]
        accelX_list=[]
        accelY_list=[]
        accelZ_list=[]
        gyroX_list=[]
        gyroY_list=[]
        gyroZ_list=[]

        for line in open(config.dataset_dir):
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
    
    def get_data(self):
        return (self.accelX_list, self.accelY_list, self.accelZ_list, 
                    self.gyroX_list, self.gyroY_list, self.gyroZ_list)
    
    
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