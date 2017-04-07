# matDL
A lightweight MATLAB deeplearning toolbox,based on gpuArray.  
One of the fastest matlab's RNN libs.
## Performance
model:A LSTM model has [1024,1024,1024] hidensizes and 10 
timestep with a 256 dims input.  
Device: i7-4710hq,GTX940m  
matDL: 60sec/epoch Keras(1.2.2,Tensorflow backend,cudnn5.1): 29sec/epoch 
## Features
High parallel Implementation.


* Concatance the weights of 4 gates to **W** and the values of **x** and **h** of every timesteps in a batch to a 3D tensor **xh**.Compute **x*W** for every timesteps of every samples in a batch at one time.
* Compute the activated values of **input,forget ,ouput gates** at one time.

OOP style
* Use `struct` type to define a **layer** class and a **model** class.Define **ff**, **bp**, **optimize** methods by using a `FunctionHandle`.  

## APIs
### Model
* A `model` is a set of `layers`,`data` and `optimizer`.
* build
    * `model=model_init(input_shape,configs ,flag,optimizer)`
    * arguments:  
        * `input_shape` : a `vector`,`[input_dim,batchsize]` or `[input_dim,timestep,batchsize]`
        * `configs` : `cell` ,configures of each layers  
        * `flag` : `bool` ,0 is predict model,1 is trrain model
        * `optimizer` : `struct` ,keywords: `opt`(type of optimizer) ,`learningrate` 
* attributes :
    * `model.input_shape`  
    * `model.output_shape`
    * `model.batchsize`
    * `model.configs`
    * `model.flag`
    * `model.layers`
    * `model.optimizer` (if `flag`)
    * `model.loss`
* methods:
    * private:
        * `model.eval_loss=@(outputlayer,y_true)eval_loss(outputlayer,y_true)`
        * `model.optimize=@(layer,batch,epoch)layer_optimize(layer,optimizer,batch,epoch)`
    * public:
        * `model.train=@(x,y,nb_epoch,verbose,filename)model_train(model,x,y,nb_epoch,verbose,filename)`
            * `model=model.train(x,y,nb_epoch,verbose,filename)`  
                * arguments:
                    * `x`:input,shape:[dim,timestep,nb_samples],or [dim,nb_samples]  
                    * `y`:targets  
                    * `nb_epoch`: how many epochs you want to train
                    * `verbose` :0,1,2,3,0 means no waitbar an figure,1 means showing waitbar only,2 means showing waitbar and plotting figures every epoch,3 means  showing waitbar and plotting figures every epoch an batch.   
        * `model.predict=@(x)model_predict(model,x)`
            * `y=model.predict(x)`  
        * `**TODO:evaluate**`
        * `model.save=@(filename)model_save(model,filename)`  
            *  `model.save(filename)`    
            * Save layers weigths and configs to a`.mat` file.
* reload:  
    * `model=model_load(minimodel,batch_size,flag,optimizer)`   
        * `minimodel` is the minimodel saved by `model.save()`,can be a `struct` variable or a `string` of filename.  
* **example**: 
x=rand(100,10,3200,'single','gpuArray');   
y=(zeros(512,10,3200'single','gpuArray'));  
y(1,:,:)=1;  
%% Define a model which has 2 lstm layers with 512 hiddenunits,and a timedistrbuted dense layer with 512 hiddenunits  
input_shape=[100,10,64];%input dim is 100,timestep is 10,batchsize is 64  
hiddensizes=[512,512,512];  
for l=1:length(hiddensize)  
    configs{l}.type='lstm';  
    configs{l}.hiddensize=hiddensize(l);  
    configs{l}.return_sequence=1;  
end  
configs{l+1}.type='activation';  
configs{l+1}.act_fun='softmax';  
configs{l+1}.loss='categorical_cross_entropy';  
optimizer.learningrate=0.1;  
optimizer.momentum=0.2;  
optimizer.opt='sgd';
model=model_init(input_shape,configs,1,optimizer);  
%% Train the model  
model=model_train(model,x,y,nb_epoch,3,'example/minimodel_f.mat');  
    

### Layers
#### Layer class: 
* attributes:  
    * `type` : `string`,type of the layer,available types:`input`,`dense`,`lstm`,`activation`  
    * `prelayer_type` : `string`,type of the previous layer,available types:`input`,`dense`,`lstm`,`activation`
    * `trainable` : `bool`,is the layer trainable
    * `flag` : train model or predict model  
    * `configs` :configures of the layer  
    * `input_shape` : `vector`,`[input_dim,batchsize]` or `[input_dim,timestep,batchsize]`
    * `output_shape` : `vector`,`[hiddensize,batchsize]`or`[hiddensize,timestep,batchsize]`
    * `batch` : `int`,how many batches have been passed
    * `epoch` : same to `batch`
* methods:  
    * `layer=**layer_init(prelayer,loss,kwgrs)`
        * Built and init a layer.If the layer is a `input` layer,`prelayer` argument should be `input_shape`
    * `layer=layer.ff(layer,prelayer)`
    * `layer=layer.bp(layer,nextlayer)`  
    ##### LSTM layer(layer)  
        * `layer=lstm_init_gpu(prelayer,hiddensize,return_sequence,flag,loss)`
        * A LSTM(**Long-Short Term Memory unit - Hochreiter 1997**) layer,see [there]:http://deeplearning.net/tutorial/lstm.html for a step-by-step description of the algorithm.
            * aviliable configures:
                * `config.hiddensize` : `int`(`double`),number of hidden units(output dim)
                * `config.return_sequence` :`bool`(`double`),return sequences or not.if `return_sequences`,output will be a 3D tensor with shape (hiddensize,timestep,batchsize). Else ,a 2D tensor with shape (hiddensize,batchsize). 
                * `config.loss` : `string`,type of loss function.Optional,only be used if the layer is an ouput layer.
                * **example**
                

	 
