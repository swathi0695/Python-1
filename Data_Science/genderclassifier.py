from sklearn import tree

#Sample data
#[height, weight, shoe size]
X = [[181,80,44], [177,70,43], [160,60,38], [187,78,46]]
#Gender
Y = ['male', 'female', 'male', 'male']


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
prediction = clf.predict([[177,70,43]])
print(prediction)
