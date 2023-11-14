import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('train.csv')
age1 = df[df['Pclass'] == 1]['Age'].median()
age2 = df[df['Pclass'] == 2]['Age'].median()
age3 = df[df['Pclass'] == 3]['Age'].median()

def change_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return age1
        elif row['Pclass'] == 2:
            return age2
        else:
            return age3
    else:
        return row['Age']
    
def change_sex(row):
    if row['Sex'] == 'male':
        return 1
    return 0

#ALL CHANGES ON NUMBERS
df['Embarked'].fillna('S', inplace=True)
df['Age'] = df.apply(change_age, axis=1)
df['Sex'] = df.apply(change_sex, axis=1)
df[list(pd.get_dummies(df["Embarked"]).columns)] = pd.get_dummies(df["Embarked"])
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

print(df.info())
x = df.drop('Survived', axis = 1)
y = df['Survived']

x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(accuracy_score(y_test, y_pred) * 100)
