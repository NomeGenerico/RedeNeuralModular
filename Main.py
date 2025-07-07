import numpy as np
import pickle
import os
import struct
# from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter





class Neuron:
    def __init__(self,PLConnections, CLConnections, Iid):

        CLw = []     #np.array()
        PLw = []     #np.array()
        CLidl = []
        PLidl = []
        for i in range(CLConnections):
            id = i
            CLw.append(np.random.uniform(-0.8,0.8))
            CLidl.append(id)
        
        for i in range(PLConnections):
            id = i
            PLw.append(np.random.rand())
            PLidl.append(id)

        self.CombinedW = np.concatenate([PLw,CLw])
        self.CombinedIDl = PLidl.copy()
        for i in CLidl:
            self.CombinedIDl.append(i+PLConnections)

        self.id = Iid

        
        pass
    # Desc: Self evident, dont you think

    # Weights is a list of tuples, id (index of the neuron): Weight (weight of the sinapse)
    # weights could include a spcecific set exclusive for inputs and outputs. As it is a "rnn" (maybe) outputs are not alwais useful or
    #  intended by the model, and since its not linear in space maybe we could have a special neuron that chooses when to stop and say this is my answer.   


    # Activation params # todo: chose a better data types, there might be no need for high precision 

    # Modulus = False     # takes the modulo of the activation function, alows for neuron with diferent functions
    # M_scale = 1         # multiplies de derivative of TANH.     f(a*x)
    # B_scale = 0         # translate TANH horizontaly            f(x + b)
    # C_scale = 0         # translate TANH verticaly              f(x) + c
    # Threshold if using 1 bit, like human neurons determines minimum value to round to 1. 
    
class NeuronMap:

    # This stores the activation level of all neurons at a point in given time.
    # It is used during inferrence of the next time step and might be used during training via 
    # small world network optimization, as pruning of synapses is hevily encuraged for performance 
    # reasons (or not, maybe using a fixed size could be better. real neurons also have a limited
    #  number of connection they can have)


    # Output of the activation of each neuron will be stored acording to some id (index) this id is implicit,
    # a neuron does not know who it is, just who it relates to;

    def __init__(self, nNeuron, PLConnections, CLConnections, functiontype):
            
        self.FunctionType = functiontype
        self.NMap = {}
        for i in range(nNeuron):
            self.NMap[i] = Neuron(PLConnections,CLConnections, i)

    def __getitem__(self,key):
        return self.NMap[key]
    
    def __setitem__(self, key, value):
        self.NMap[key] = value

    def copy(self):
        # retorna uma cópia profunda do mapa
        new_map = NeuronMap(0, 0)
        new_map.NMap = self.NMap.copy()
        return new_map
    
    def items(self):
        return self.NMap.items()
    
    def keys(self):
        return self.NMap.keys()
    
    def values(self):
        return self.NMap.values()
    
    def __len__(self):
        return len(self.NMap)
    

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    

    def ReLU(self,x):
        return max(0,x)

    def ActivationFunction(self, neuron, reducedPGmap):

        preActiv = np.array(reducedPGmap) * np.array(neuron.CombinedW)   # PreActiv[i] = PGMap[i] * w[i]

        if self.FunctionType == 1:
            return self.ReLU(np.sum(preActiv))
        if self.FunctionType == 2:
            return self.sigmoid(np.sum(preActiv))  
        
class GradientMap:

    def __init__(self, nNeuron, nConnections = 0):
            
        self.GMap = {}
        for i in range(nNeuron):
            self.GMap[i] = 0.0

    def __getitem__(self,key):
        return self.GMap[key]

    def __setitem__(self, key, value):
        self.GMap[key] = value
    
    def copy(self):
        # retorna uma cópia profunda do mapa
        new_map = GradientMap(0, 0)
        new_map.GMap = self.GMap.copy()
        return new_map
    
    def items(self):
        return self.GMap.items()
    
    def keys(self):
        return self.GMap.keys()
    
    def values(self):
        return self.GMap.values()
    
    def __len__(self):
        return len(self.GMap)
    
class LayeredRNNModel:

    def __init__(self,FilePath = False):
        
        if(FilePath != False):  # Load Model
            self.LoadModel(FilePath)
        else:    
            #Create New Model
            self.numberOfLayers = 3 # input + Hiden + Output.  min = 2
            self.inputSize = 784
            self.outputSize = 10  # OutputSize + 1 stop neuron

            self.nINeuron = 100
            self.nHNeuron = 50
            self.outputSize = 10 # 10 digits + a "Im done Processing" neuron
            self.nONeuron = 10

            #NeuronMaps are Model Specific
            self.INMap = NeuronMap(self.nINeuron,self.inputSize,self.nINeuron, 1)                 #NeuronMap of input layer
            self.HNMap = NeuronMap(self.nHNeuron,self.nINeuron,self.nHNeuron, 1)                 #NeuronMap of hiden layer
            self.ONMap = NeuronMap(self.nONeuron,self.nHNeuron,self.nONeuron, 2)                 #NeuronMap of output layer
            
            self.NMapList = [self.INMap,self.HNMap,self.ONMap]
            self.BackwardsNMapList = [self.INMap,self.HNMap]

            #Model History for training:
            # list index is the tick and value is a list of [inputMap, inputGradMap, hGradMap, outGradMap]
            self.History = []

            # ForwardsMap Maps a neuron To a list of neurons that are connected to it and their weights:
            self.LayerForwardsMap = [{} for _ in range(len(self.NMapList) - 1)]

            for layer_idx in range(len(self.LayerForwardsMap)):  # entre layer i e layer i+1
                curr_layer = self.NMapList[layer_idx]
                next_layer = self.NMapList[layer_idx + 1]

                forward_map = self.LayerForwardsMap[layer_idx]

                for neuron in next_layer.NMap.values():  # cada neurônio da próxima camada
                    for idx, input_id in enumerate(neuron.CombinedIDl):
                        weight = neuron.CombinedW[idx]
                        if input_id not in forward_map:
                            forward_map[input_id] = []
                        forward_map[input_id].append({neuron.id: weight})

            self.LayerList = [self.INMap, self.HNMap, self.ONMap]
            self.GradList = []

    def Softmax(self, x_vec):
        exp_vals = np.exp(x_vec - np.max(x_vec))  # evitar overflow numérico
        return exp_vals / np.sum(exp_vals)
   
    def LocalBP(self, Layer, Neuron, GradList, learning_rate):

        PLGradMap = GradList[Layer - 1]
        CLGradMap = GradList[Layer]
        NLGradmap = GradList[Layer + 1]

        if Layer != len(GradList) - 2:

            AverageWeightedDependentActivation = 0
            TotalWeight = 0
            #print(Layer, Neuron.id)
            for IdWPair in self.LayerForwardsMap[Layer - 1][Neuron.id]:
                for ID in IdWPair.keys():
                    AverageWeightedDependentActivation += NLGradmap.GMap[ID] * IdWPair[ID]
                    TotalWeight += IdWPair[ID]

            AverageWeightedDependentActivation /= TotalWeight  # TODO Consider cliping weights to [-1,1] or smh

        else: 
            #print(Layer, Neuron.id)
            AverageWeightedDependentActivation = NLGradmap.GMap[Neuron.id]
        
        Error = AverageWeightedDependentActivation - CLGradMap[Neuron.id]

        for idx, input_id in enumerate(Neuron.CombinedIDl):
            input_value = PLGradMap[input_id] if input_id < len(PLGradMap) else 0
            Neuron.CombinedW[idx] -= learning_rate * Error * input_value



        return

    def TickTimeLayer(self, Layer, PGradList, learning_rate=0):
        NMap = self.LayerList[Layer - 1]
        PLGradMap = self.GradList[Layer - 1]
        CLGradMap = self.GradList[Layer]
        PastCLGradMap = PGradList[Layer - 1]

        CombGradMap = {}
        for id in PLGradMap.keys():
            CombGradMap[id] = PLGradMap[id]
        for id in PastCLGradMap.keys():
            CombGradMap[id + len(PLGradMap)] = PastCLGradMap[id]

        def process_neuron(id, neuron):
            reducedCombGradMap = [CombGradMap[i] for i in neuron.CombinedIDl]
            activation = NMap.ActivationFunction(neuron, reducedCombGradMap) / len(NMap)
            CLGradMap[id] = activation
            if learning_rate != 0:
                self.LocalBP(Layer, neuron, self.GradList, learning_rate)

        # ☑️ Aqui sim a execução acontece
        max_threads = 12
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(process_neuron, id, neuron) for id, neuron in NMap.items()]
            for future in futures:
                future.result()


            return "placeholder"

    def Inference(self,inferenceSteps,INPUT  = False, LOG = True, learning_rate=0, ExpectedOutput = False):

         # GradMaps are inference Specific
        inputMap = GradientMap(self.inputSize)           #GradMap of the inputs
        inputGradMap = GradientMap(self.nINeuron)       #GradMap of the input layer
        hGradMap = GradientMap(self.nHNeuron)           #GradMap of the hiden Layer
        outGradMap = GradientMap(self.nONeuron)             #GradMap of the output layer
        TrainingMap = GradientMap(self.nONeuron)    # Expected output Based on training data



        self.GradList = [inputMap, inputGradMap, hGradMap, outGradMap, TrainingMap]

        if LOG:
            a = []
            for i in self.INMap.NMap:
                a.append(i)

            b = []
            for i in self.HNMap.NMap:
                b.append(i)
    
            c = []
            for i in self.ONMap.NMap:
                c.append(i)

            print(a)
            Nlist = (self.INMap.NMap.values()) # list of {id:neuron}
            print(type(Nlist))
            alist = []
            for i in Nlist:
                alist.append(i)
            print(alist[0].CombinedIDl)
            print(alist[0].CombinedW)
            print(b)
            print(c)

        image_pixels = []
        if INPUT is False:
            for i in range(self.inputSize):
                inputMap[i] = np.random.rand()
        else:
            for i in range(len(INPUT)):
                inputMap[i] = INPUT[i]

        if ExpectedOutput is False:
            for i in range(self.outputSize):
                TrainingMap[i] = np.random.rand()
        else:
            for i in range(self.outputSize):
                TrainingMap[i] = ExpectedOutput[i]


    #   uses a theread pool for the neurons and layers
        for i in range(inferenceSteps):
        # update inputmap, if input changed
            PinputGradMap = inputGradMap.copy()
            PhgradMap = hGradMap.copy()
            PoutMap = outGradMap.copy() 
        
            PGradList = [PinputGradMap, PhgradMap, PoutMap]

        # Multitreading Logic{


        # Importatnt: TickTime Funcition can be made to edit inputGradMap directly instead of returning a value to be copied. The P...Maps alow for this. TODO
            #self.TickTimeLayer(self.INMap, inputMap, PinputGradMap, inputGradMap, 1 , 1)
            self.TickTimeLayer(1, PGradList, learning_rate)

            #self.TickTimeLayer(self.HNMap, PinputGradMap, PhgradMap, hGradMap, 1 , 2)
            self.TickTimeLayer(2, PGradList, learning_rate)

            #self.TickTimeLayer(self.ONMap, PhgradMap, PoutMap, outGradMap, 1 , 3)
            self.TickTimeLayer(3, PGradList, learning_rate)

        #}
            if LOG:
                print("Step", i)
                print("inputMap",inputMap.GMap.values())
                print("inputGradMap:", inputGradMap.GMap.values())
                print("hGradMap:", hGradMap.GMap.values())
                print("outGradMap:", outGradMap.GMap.values())
        # save GMaps if needed for training

            self.History.append([inputMap, inputGradMap, hGradMap, outGradMap])

        out_values = np.array(list(outGradMap.values()))
        softmax_output = self.Softmax(out_values)
        for i, val in enumerate(softmax_output):
            outGradMap[i] = val

        return outGradMap
    
    def SaveModel(self, filename="saved_model.pkl"):
        model_data = {
            "NMapList": [],
            "inputSize": self.inputSize,
            "nINeuron": self.nINeuron,
            "nHNeuron": self.nHNeuron,
            "nONeuron": self.nONeuron,
        }

        for nmap in self.NMapList:
            layer_data = []
            for neuron in nmap.NMap.values():
                neuron_data = {
                    "CombinedW": neuron.CombinedW.tolist(),
                    "CombinedIDl": neuron.CombinedIDl,
                    "id": neuron.id
                }
                layer_data.append(neuron_data)
            model_data["NMapList"].append(layer_data)

        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

    def LoadModel(self, filename="saved_model.pkl"):
        if not os.path.exists(filename):
            print(f"Arquivo {filename} não encontrado.")
            return

        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        self.inputSize = model_data["inputSize"]
        self.nINeuron = model_data["nINeuron"]
        self.nHNeuron = model_data["nHNeuron"]
        self.nONeuron = model_data["nONeuron"]

        self.NMapList = []
        for layer_data in model_data["NMapList"]:
            nmap = NeuronMap(0, 0, 0, 1)
            nmap.NMap = {}
            for neuron_data in layer_data:
                neuron = Neuron(0, 0, neuron_data["id"])
                neuron.CombinedW = np.array(neuron_data["CombinedW"])
                neuron.CombinedIDl = neuron_data["CombinedIDl"]
                nmap.NMap[neuron.id] = neuron
            self.NMapList.append(nmap)

        # Reconfigura os elementos derivados
        self.INMap, self.HNMap, self.ONMap = self.NMapList
        self.LayerList = [self.INMap, self.HNMap, self.ONMap]

        # Reconstroi LayerForwardsMap
        self.LayerForwardsMap = [{} for _ in range(len(self.NMapList) - 1)]
        for layer_idx in range(len(self.LayerForwardsMap)):
            curr_layer = self.NMapList[layer_idx]
            next_layer = self.NMapList[layer_idx + 1]
            forward_map = self.LayerForwardsMap[layer_idx]

            for neuron in next_layer.NMap.values():
                for idx, input_id in enumerate(neuron.CombinedIDl):
                    weight = neuron.CombinedW[idx]
                    if input_id not in forward_map:
                        forward_map[input_id] = []
                    forward_map[input_id].append({neuron.id: weight})

    def Train(self, DATASET, inferenceSteps=10, learning_rate=1, Epochs=1):
        x_data, y_labels = DATASET
        total_samples = len(x_data) * Epochs

        start_time = time.time()

        for epoch in range(Epochs):
            for i in range(len(x_data)):
                idx = epoch * len(x_data) + i
                x_input = x_data[i]
                y_target = y_labels[i]

                # Inference + aprendizado
                output = self.Inference(inferenceSteps, x_input, False, learning_rate, y_target)

                # A cada 10 amostras, mostra tempo e progresso
                if idx % 10 == 0 and idx > 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / idx
                    remaining = total_samples - idx
                    eta = remaining * avg_time

                    percent = 100 * idx / total_samples
                    print(f"[{percent:.1f}%] {idx}/{total_samples} amostras — "
                        f"Tempo: {elapsed:.1f}s — ETA: {eta:.1f}s")
                    
                    self.SaveModel("mnist_model.pkl")
        total_time = time.time() - start_time
        print(f"\n✅ Treino concluído em {total_time:.1f} segundos.")

    def predict(self, image_input, inferenceSteps=10):
        output_map = self.Inference(inferenceSteps, image_input, LOG=False, learning_rate=0)
        output_values = np.array(list(output_map.values()))
        return np.argmax(output_values), output_values

def load_mnist_from_folder(folder_path, kind='train', n_samples=None):
    labels_path = os.path.join(folder_path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(folder_path, f'{kind}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
        if n_samples:
            labels = labels[:n_samples]

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8)

        if n_samples:
            images = images[:n_samples * 784]

        images = images.reshape(len(labels), 784)
        images = images.astype(np.float32) / 255.0  # normaliza [0,1]

    # One-hot encode
    one_hot_labels = np.zeros((len(labels), 10), dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot_labels[i, label] = 1.0

    return images, one_hot_labels

def TestModel(model, test_dataset, inferenceSteps=10):
    x_test, y_test = test_dataset
    correct = 0
    total = len(x_test)
    predicted_list = []

    for i in range(total):
        x_input = x_test[i]
        expected_label = np.argmax(y_test[i])  # one-hot decode

        output_map = model.Inference(inferenceSteps, x_input, LOG=False)
        output_values = list(output_map.items())
        predicted_label = max(output_values, key=lambda x: x[1])[0]
        predicted_list.append(predicted_label)

        if predicted_label == expected_label:
            correct += 1

        if i % 1000 == 0:
            print(f"[{i}/{total}] Exemplos testados")

    print("\nDistribuição das predições:")
    print(Counter(predicted_list))

    accuracy = correct / total
    print(f"\n🎯 Acurácia no conjunto de teste: {accuracy * 100:.6f}%")






Model = LayeredRNNModel()
inferenceSteps = 10

# Model.LoadModel()
# output = Model.Inference(inferenceSteps, False, False, 1)
# Model.SaveModel()
# print(output.values())
# print(len(Model.History))

TRAININGDATASET = load_mnist_from_folder("DATASETS/MINST","train",500)
TESTINGDATASET = load_mnist_from_folder("DATASETS/MINST","t10k",500)
# print(TRAININGDATASET[0][0])
# print(TESTINGDATASET[1][0])
# Model.Train(TRAININGDATASET)  # só 500 para testar performance
Model.LoadModel()
Model.Train(TRAININGDATASET)
TestModel(Model, TRAININGDATASET)
