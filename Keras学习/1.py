from keras.layers import Dense, np
from keras.models import Sequential
import matplotlib.pyplot as plt


X = np.linspace(-1,1,200)#在指定的间隔内返回均匀间隔的数字。
np.random.shuffle(X) #打乱X数据顺序
Y=0.5*X+2+np.random.normal(0,0.05,(200,))

plt.scatter(X,Y)
plt.show()

x_train,y_train=X[:160],Y[:160]
x_test,y_test=X[160:],Y[160:]

#
#
model = Sequential()
model.add(Dense(output_dim=1,input_dim=1))

model.compile(loss='mse',optimizer='sgd')

print('-----')
for sep in range(500):
    cost =  model.train_on_batch(x_train,y_train)
    if sep%100==0:
        print('train cost:',cost)

print('test-----')

cost = model.evaluate(x_test,y_test)

print('test cost:',cost)
w,b = model.layers[0].get_weights()
print('weights=',w,'\nbiascs=',b)

y_pred = model.predict(x_test)
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,color='#00161a')
plt.show()





