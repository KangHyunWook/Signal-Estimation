ma_w_size=1
lw_size=1
tri_w_size=1

python main.py --dataset_dir "C:\workspace\NYUAD\raw_imu_pattern1.csv" --axis 0 --smoothing --ma_window $ma_w_size --long_window_size $lw_size --triangle_filter --triangle_w_size $tri_w_size

while [ $ma_w_size -le 30 ]
	do
	while [ $lw_size -le 30 ]
		do
		while [ $tri_w_size -le 30 ]	
			do
				((tri_w_size=tri_w_size+1))
				python main.py --dataset_dir "C:\workspace\NYUAD\raw_imu_pattern1.csv" --axis 0 --smoothing --ma_window $ma_w_size --long_window_size $lw_size --triangle_filter --triangle_w_size $tri_w_size --w_mode 'a'
			done
			((lw_size=lw_size+1))
			tri_w_size=0
		done
		((ma_w_size=ma_w_size+1))
		lw_size=1
	done