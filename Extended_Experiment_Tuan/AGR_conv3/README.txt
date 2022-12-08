- my_model is defined new -> all layers except for dense layers are set to trainable true
- dataset fairface is preprocessed -> image labels are saved in new format similar to UTK_Face dataset
- test1_ft.py saved dataset as .mat for training 
- fit doesn't work because Fairface has not attribute age from 0-117.
- UTK Face has age as a int from 0-116, fairface gives only ranges
- UTK Face transformation to Fairface ranges

-  Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 64, 64, 3)]  0           []                               
                                                                                                  
 batch_normalization (BatchNorm  (None, 64, 64, 3)   12          ['input_1[0][0]']                
 alization)                                                                                       
                                                                                                  
 conv2d (Conv2D)                (None, 64, 64, 32)   896         ['batch_normalization[0][0]']    
                                                                                                  
 conv2d_1 (Conv2D)              (None, 64, 64, 32)   9248        ['conv2d[0][0]']                 
                                                                                                  
 max_pooling2d (MaxPooling2D)   (None, 32, 32, 32)   0           ['conv2d_1[0][0]']               
                                                                                                  
 batch_normalization_1 (BatchNo  (None, 32, 32, 32)  128         ['max_pooling2d[0][0]']          
 rmalization)                                                                                     
                                                                                                  
 conv2d_2 (Conv2D)              (None, 32, 32, 64)   18496       ['batch_normalization_1[0][0]']  
                                                                                                  
 conv2d_3 (Conv2D)              (None, 32, 32, 64)   36928       ['conv2d_2[0][0]']               
                                                                                                  
 max_pooling2d_1 (MaxPooling2D)  (None, 16, 16, 64)  0           ['conv2d_3[0][0]']               
                                                                                                  
 batch_normalization_2 (BatchNo  (None, 16, 16, 64)  256         ['max_pooling2d_1[0][0]']        
 rmalization)                                                                                     
                                                                                                  
 conv2d_4 (Conv2D)              (None, 16, 16, 128)  73856       ['batch_normalization_2[0][0]']  
                                                                                                  
 conv2d_5 (Conv2D)              (None, 16, 16, 128)  147584      ['conv2d_4[0][0]']               
                                                                                                  
 conv2d_6 (Conv2D)              (None, 16, 16, 128)  147584      ['conv2d_5[0][0]']               
                                                                                                  
 max_pooling2d_2 (MaxPooling2D)  (None, 8, 8, 128)   0           ['conv2d_6[0][0]']               
                                                                                                  
 batch_normalization_3 (BatchNo  (None, 8, 8, 128)   512         ['max_pooling2d_2[0][0]']        
 rmalization)                                                                                     
                                                                                                  
 conv2d_7 (Conv2D)              (None, 8, 8, 256)    295168      ['batch_normalization_3[0][0]']  
                                                                                                  
 conv2d_8 (Conv2D)              (None, 8, 8, 256)    590080      ['conv2d_7[0][0]']               
                                                                                                  
 conv2d_9 (Conv2D)              (None, 8, 8, 256)    590080      ['conv2d_8[0][0]']               
                                                                                                  
 conv2d_10 (Conv2D)             (None, 8, 8, 256)    590080      ['conv2d_9[0][0]']               
                                                                                                  
 batch_normalization_4 (BatchNo  (None, 8, 8, 256)   1024        ['conv2d_10[0][0]']              
 rmalization)                                                                                     
                                                                                                  
 conv2d_11 (Conv2D)             (None, 8, 8, 512)    1180160     ['batch_normalization_4[0][0]']  
                                                                                                  
 conv2d_12 (Conv2D)             (None, 8, 8, 512)    2359808     ['conv2d_11[0][0]']              
                                                                                                  
 max_pooling2d_3 (MaxPooling2D)  (None, 4, 4, 512)   0           ['conv2d_12[0][0]']              
                                                                                                  
 flatten (Flatten)              (None, 8192)         0           ['max_pooling2d_3[0][0]']        
                                                                                                  
 dense (Dense)                  (None, 2)            16386       ['flatten[0][0]']                
                                                                                                  
 dense_1 (Dense)                (None, 9)            73737       ['flatten[0][0]']                
                                                                                                  
 dense_2 (Dense)                (None, 5)            40965       ['flatten[0][0]']                
                                                                                                  
==================================================================================================
Total params: 6,172,988
Trainable params: 6,172,022
Non-trainable params: 966

- training time: 3:45 per epoch