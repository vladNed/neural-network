from sklearn.datasets import make_moons

#Generate Dataset
X_train, Y_label = make_moons(n_samples=10, noise=0.1)

#Create model
model = NeuralNetwork(X_train.shape[1],10,learning_rate=2e-3)

#Train model
model.train_network(X_train,Y_label,150)

#For loading weights
#model.save_model()
#model.load_weights('nn_weights.pickle')

for i,_ in enumerate(X_train):

    print('Sample is ',X_train[i])
    print('Label is ',Y_label[i])
    p = model.predict(X_train[i])
    print('Prediction is %s' % p)
    if p != Y_label[i]:
        print("WRONG\n")
    else:
        print("CORRECT\n")
