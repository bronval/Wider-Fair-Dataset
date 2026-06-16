import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dlib
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def load_images_from_folder(folder, dataframe, targets, max_images_per_class=100):
    excluded_classes=['Undetermined', 'Other', 'Middle Eastern']
    df = pd.read_csv(dataframe)

    df['Race_Gender'] = df['Race'] + "_" + df['Gender']
    df = df.groupby('filename').filter(
        lambda group: not ((group['Gender'] == 'Undetermined').any() or 
                           (group['Race'].isin(excluded_classes)).any() or 
                           (group['Valid'] == False).any())
    )
    df = df[
        (df['blur'] == 0) &
        (df['expression'] == 0) &
        (df['illumination'] == 0) &
        (df['invalid'] == 0) &
        (df['occlusion'] == 0) &
        (df['pose'] == 0) &
        (df['Valid'] == True) &
        (df['Race_Gender'].isin(targets))
    ]
    print(len(df))
    images, labels = [], []
    counter = {target: 0 for target in targets}

    for _, row in df.iterrows():
        target = row['Race_Gender']
        if counter[target] >= max_images_per_class:
            continue
        
        img_path = os.path.join(folder, row['filename'])
        img = cv2.imread(img_path)

        if img is not None:
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = img[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            images.append(face_rgb)
            labels.append(target)
            counter[target] += 1
    print(len(images))
    return images, labels


def get_face_embeddings(images, labels):
    ### Come directly from DLIB REPO ###
    model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    embeddings = []
    valid_labels = []

    for i, img in enumerate(images):
        dets = detector(img, 1)
        if len(dets) == 0:
            continue
        shape = sp(img, dets[0])
        face_descriptor = model.compute_face_descriptor(img, shape)
        embeddings.append(np.array(face_descriptor))
        valid_labels.append(labels[i])  

    embedding_df = save_embeddings_to_dataframe(embeddings, valid_labels)
    embedding_df.to_csv("face_embeddings_test.csv", index=False)
    return np.array(embeddings), valid_labels

def reduce_with_tsne(embeddings):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    return reduced

def plot_embeddings(embeddings_2d, labels):
    races = [label.split('_')[0] for label in labels]
    sexes = [label.split('_')[1] for label in labels]
    markers = {'Male': 'o', 'Female': 'X'}
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=races,
        style=sexes,
        palette='Set1',
        markers=markers, 
        s=50,
        edgecolor='k'
    )
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Ethnicity / Sex', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def save_embeddings_to_dataframe(embeddings, labels):
    races = [label.split('_')[0] for label in labels]
    sexes = [label.split('_')[1] for label in labels]

    df = pd.DataFrame(embeddings)
    df['Race'] = races
    df['Sex'] = sexes
    df['Race_Gender'] = labels

    return df

def evaluate_knn_by_label_type(
    folder="subset_images",
    dataset="filtered_dataset_x1y1x2y2.csv",
    embedding_csv_path="face_embeddings.csv",
    ethnicities=["White", "Black", "Indian", "Asian"],
    sex=["Male", "Female"],
    max_images_per_class=2000
):  
    viz_folder = "viz_thesis"
    part = "Annotation_process"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    viz_folder = os.path.join(project_root, "viz_thesis", part)
    os.makedirs(viz_folder, exist_ok=True)

    targets = [f"{e}_{s}" for e in ethnicities for s in sex]
    subfoler_image_path = os.path.join(project_root, folder)
    if not os.path.exists(embedding_csv_path):
        images, labels = load_images_from_folder(subfoler_image_path, dataset, targets, max_images_per_class)
        embeddings, _ = get_face_embeddings(images, labels)
    else:
        df_emb = pd.read_csv(embedding_csv_path)
        embeddings = df_emb.iloc[:, :-3].values

    for label_type in ["Race", "Sex"]:
        print(f"\n=== Classification Report for {label_type} ===")

        labels = df_emb[label_type]
        class_names = sorted(labels.unique())

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )

        knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        print(classification_report(y_test, y_pred, target_names=class_names))

        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        accuracy = np.trace(cm) / np.sum(cm)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        label_name = "Ethnicity" if label_type == "Race" else "Sex"
        plt.title(f"Confusion Matrix (KNN) - {label_name} | Accuracy: {accuracy:.2%}")
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        embeddings_folder = os.path.join(viz_folder, 'embeddings')
        os.makedirs(embeddings_folder, exist_ok=True)
        filename = f'confusion_matrix_knn_{label_name.lower()}.png'
        save_path = os.path.join(embeddings_folder, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

        print(f"Confusion matrix saved as '{filename}'")


if __name__ == "__main__":
    
    evaluate_knn_by_label_type()