import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score
import graphviz

# 1. Відкрити та зчитати наданий файл з даними.
df = pd.read_csv('WQ-R.csv', sep=';')

# 2. Визначити та вивести кількість записів та кількість полів у завантаженому наборі даних.
num_of_rows = len(df)
num_of_columns = len(df.columns)
print('Кількість записів:', num_of_rows)
print('Кількість полів:', num_of_columns)

# 3. Вивести перші 10 записів набору даних.
print(df.head(10))

# 4. Розділити набір даних на навчальну (тренувальну) та тестову вибірки.
df_train, df_test = train_test_split(df, train_size=0.8, random_state=1)

# 5. Використовуючи відповідні функції бібліотеки scikit-learn, збудувати класифікаційну модель дерева прийняття рішень глибини 5 та навчити її на тренувальній вибірці.
x_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
x_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

model_g = DecisionTreeClassifier(max_depth=5, criterion='gini')
model_g.fit(x_train, y_train)

model_e = DecisionTreeClassifier(max_depth=5, criterion='entropy')
model_e.fit(x_train, y_train)

# 6. Представити графічно побудоване дерево за допомогою бібліотеки graphviz.
img_g = export_graphviz(
    model_g,
    feature_names=x_train.columns,
    class_names=list(map(str, y_train.unique())),
    rounded=True,
    filled=True
)
graph_g = graphviz.Source(img_g)
graph_g.render('tree_gini', format='png')

img_e = export_graphviz(
    model_e,
    feature_names=x_train.columns,
    class_names=list(map(str, y_train.unique())),
    rounded=True,
    filled=True
)
graph_e = graphviz.Source(img_e)
graph_e.render('tree_entropy', format='png')

# 7. Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки. Представити результати роботи моделі на тестовій вибірці графічно. Порівняти результати, отриманні при застосуванні різних критеріїв розщеплення: інформаційний приріст на основі ентропії чи неоднорідності Джині.

def metrics(model, x, y):
    predictions = model.predict(x)
    return {
        'Accuracy': accuracy_score(y, predictions),
        'Precision': precision_score(y, predictions, average='weighted'),
        'Recall': recall_score(y, predictions, average='weighted'),
        'F1-Score': f1_score(y, predictions, average='weighted'),
        'MCC': matthews_corrcoef(y, predictions),
        'Balanced Accuracy': balanced_accuracy_score(y, predictions),
    }

test_g = metrics(model_g, x_test, y_test)
test_e = metrics(model_e, x_test, y_test)

train_g = metrics(model_g, x_train, y_train)
train_e = metrics(model_e, x_train, y_train)

fig, axs = plt.subplots(3, 2, figsize=(12, 15))
metrics_names = list(test_g.keys())
for i, metric in enumerate(metrics_names):
    axs[i // 2, i % 2].plot([0, 1], [test_g[metric]]*2, label='test_gini')
    axs[i // 2, i % 2].plot([0, 1], [test_e[metric]]*2, label='test_entropy')
    axs[i // 2, i % 2].plot([0, 1], [train_g[metric]]*2, label='train_gini', linestyle='--')
    axs[i // 2, i % 2].plot([0, 1], [train_e[metric]]*2, label='train_entropy', linestyle='--')
    axs[i // 2, i % 2].set_title(metric)
    axs[i // 2, i % 2].legend()
plt.tight_layout()
plt.show()

plt.bar(train_g.keys(), train_g.values())
plt.title('Train gini metrics values')
plt.show()

plt.bar(train_e.keys(), train_e.values())
plt.title('Train entropy metrics values')
plt.show()

# 8. З’ясувати вплив глибини дерева та мінімальної кількості елементів в листі дерева на результати класифікації. Результати представити графічно.

min_samples_leaf_values = range(1, 42)
balanced_accuracies_test = []
balanced_accuracies_train = []

for value in min_samples_leaf_values:
    model = DecisionTreeClassifier(min_samples_leaf=value, random_state=1)
    model.fit(x_train, y_train)
    balanced_accuracies_train.append(balanced_accuracy_score(y_train, model.predict(x_train)))
    balanced_accuracies_test.append(balanced_accuracy_score(y_test, model.predict(x_test)))

plt.plot(min_samples_leaf_values, balanced_accuracies_test, label='test')
plt.plot(min_samples_leaf_values, balanced_accuracies_train, label='train')
plt.xlabel('Min Samples Leaf')
plt.ylabel('Balanced Accuracy')
plt.legend()
plt.title('Bплив мінімальної кількості елементів в листі дерева на результати класифікації.')
plt.show()

max_depth_values = range(1, 42)
balanced_accuracies_test = []
balanced_accuracies_train = []

for value in max_depth_values:
    model = DecisionTreeClassifier(max_depth=value, random_state=1)
    model.fit(x_train, y_train)
    balanced_accuracies_train.append(balanced_accuracy_score(y_train, model.predict(x_train)))
    balanced_accuracies_test.append(balanced_accuracy_score(y_test, model.predict(x_test)))

plt.plot(max_depth_values, balanced_accuracies_test, label='test')
plt.plot(max_depth_values, balanced_accuracies_train, label='train')
plt.xlabel('Max Depth')
plt.ylabel('Balanced Accuracy')
plt.legend()
plt.title('Bплив максимальної глибини дерева на результати класифікації.')
plt.show()

# 9. Навести стовпчикову діаграму важливості атрибутів, які використовувалися для класифікації (див. feature_importances_). Пояснити, яким чином – на Вашу думку – цю важливість можна підрахувати.

feature_importances_g = model_g.feature_importances_
feature_importances_e = model_e.feature_importances_

feature_names = x_train.columns

sorted_idx_g = feature_importances_g.argsort()
plt.barh(range(len(sorted_idx_g)), feature_importances_g[sorted_idx_g], align='center')
plt.yticks(range(len(sorted_idx_g)), [feature_names[i] for i in sorted_idx_g])
plt.title('Feature importances (gini)')
plt.show()

sorted_idx_e = feature_importances_e.argsort()
plt.barh(range(len(sorted_idx_e)), feature_importances_e[sorted_idx_e], align='center')
plt.yticks(range(len(sorted_idx_e)), [feature_names[i] for i in sorted_idx_e])
plt.title('Feature importances (entropy)')
plt.show()
