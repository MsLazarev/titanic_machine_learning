import pandas as pd

df = pd.read_csv('train.csv')
#print(df.info())
#print(df.tail())
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
df[list(pd.get_dummies(df["Embarked"]).columns)] = pd.get_dummies(df["Embarked"], dtype=int)
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
print(df.head())