#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 1. Bibliothek installieren
get_ipython().system('pip install opendatasets')

import opendatasets as od

# 2. Den Link zum Datensatz angeben
dataset_url = "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"

# 3. Download starten
od.download(dataset_url)


# In[3]:


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# Pfad zum heruntergeladenen Datensatz
DATA_PATH = "./cifake-real-and-ai-generated-synthetic-images"

# Datensatz-Struktur prüfen:
# train/FAKE: 50.000 KI-generierte Bilder (Stable Diffusion 1.4)
# train/REAL: 50.000 echte Fotos (aus CIFAR-10)
# test/FAKE:  10.000 KI-generierte Bilder
# test/REAL:  10.000 echte Fotos
print("Datensatz-Struktur:")
print("-" * 35)
for split in ["train", "test"]:
    for label in ["FAKE", "REAL"]:
        pfad   = os.path.join(DATA_PATH, split, label)
        anzahl = len(os.listdir(pfad))
        print(f"  {split}/{label}: {anzahl} Bilder")


# In[4]:


# Globale Parameter — einmal definiert, überall verwendet
IMG_HEIGHT = 32     # Originale Bildgroesse im Datensatz
IMG_WIDTH  = 32
BATCH_SIZE = 64     # Anzahl Bilder pro Trainingsschritt
SEED       = 42     # Fixierter Zufallsseed fuer Reproduzierbarkeit

# image_dataset_from_directory liest Ordnerstruktur automatisch:
# FAKE/ -> Label 0  (alphabetisch zuerst)
# REAL/ -> Label 1
# label_mode="binary" weil nur 2 Klassen

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_PATH, "train"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_PATH, "test"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode="binary"
)

class_names = train_ds.class_names
print("Klassen:", class_names)
# WICHTIG: FAKE=0, REAL=1 (alphabetische Reihenfolge)

# Shape eines Batches kontrollieren
for images, labels in train_ds.take(1):
    print(f"Bild-Batch Shape:  {images.shape}")   # (64, 32, 32, 3)
    print(f"Label-Batch Shape: {labels.shape}")    # (64, 1)
    print(f"Pixelbereich:      {images.numpy().min():.0f} bis {images.numpy().max():.0f}")


# In[7]:


# Visuelle Kontrolle: sehen die Bilder korrekt aus?
# Wichtig: FAKE-Bilder sind 32x32 Stable-Diffusion-Bilder,
# nicht hochaufloesende KI-Bilder

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("CIFAKE Beispielbilder aus dem Trainingsdatensatz", fontsize=13)

for images, labels in train_ds.take(1):
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(class_names[int(labels[i])], fontsize=9)
        ax.axis("off")

plt.tight_layout()
plt.show()


# In[7]:


# Bei 100.000 Bildern ist Performance-Optimierung wichtig:
#
# cache():    Bilder nach der ersten Epoche im RAM behalten
#             -> ab Epoche 2 kein Lesen von Disk mehr noetig
#
# shuffle():  Reihenfolge der Bilder mischen
#             -> verhindert, dass das Modell die Reihenfolge lernt
#
# prefetch(): Naechsten Batch vorbereiten waehrend aktueller Batch trainiert wird
#             -> GPU/CPU laufen parallel -> schneller

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Performance-Optimierung angewendet.")


# In[8]:


# CNN = Convolutional Neural Network
# Speziell fuer Bilder entwickelt: lernt lokale Muster (Kanten, Texturen, Formen)
#
# Architektur:
# Conv2D:       sucht Muster in kleinen Bildausschnitten (3x3 Pixel)
# MaxPooling2D: verkleinert die Feature-Map auf die Haelfte -> schneller, robuster
# Flatten:      2D Feature-Map -> 1D Vektor fuer Dense-Layer
# Dense:        vollvermaschte Schicht wie bei Fashion-MNIST
# sigmoid:      Output zwischen 0 und 1 (Wahrscheinlichkeit fuer REAL)

model_base = keras.Sequential([
    # Normalisierung: 0-255 -> 0.0-1.0 direkt im Modell
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # Block 1: einfache Muster (Kanten, Farben)
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(),   # 32x32 -> 16x16

    # Block 2: komplexere Muster (Formen, Texturen)
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(),   # 16x16 -> 8x8

    # Block 3: abstrakte Merkmale
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(),   # 8x8 -> 4x4

    layers.Flatten(),
    layers.Dense(128, activation="relu"),

    # 1 Output-Neuron mit sigmoid: Wert > 0.5 -> REAL, Wert < 0.5 -> FAKE
    layers.Dense(1, activation="sigmoid")
], name="model_base")

model_base.summary()

# binary_crossentropy: korrekte Loss-Funktion fuer binaere Klassifikation
model_base.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# In[9]:


# EarlyStopping: stoppt Training wenn val_accuracy sich
# 3 Epochen lang nicht verbessert -> spart Zeit, verhindert Overfitting
# restore_best_weights: laedt Gewichte der besten Epoche zurueck

history_base = model_base.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    )]
)

base_loss, base_acc = model_base.evaluate(test_ds, verbose=0)
print(f"Basismodell Test-Accuracy: {base_acc:.2%}")


# In[12]:


# Wiederverwendbare Funktion fuer beide Modelle
# Lernkurven zeigen:
# - steigt val_accuracy mit training_accuracy mit? -> kein Overfitting
# - Luecke zwischen training und validation? -> Overfitting

def plot_history(history, title="Training"):
    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochen  = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)

    ax1.plot(epochen, acc,     label="Training")
    ax1.plot(epochen, val_acc, label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochen, loss,     label="Training")
    ax2.plot(epochen, val_loss, label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoche")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Beste Validation Accuracy: {max(val_acc):.2%}  (Epoche {val_acc.index(max(val_acc))+1})")
    print(f"Finale Training Accuracy:  {acc[-1]:.2%}")

plot_history(history_base, "Basismodell — Lernkurven")


# In[13]:


# Data Augmentation: erzeugt kuenstlich mehr Trainingsdaten
# durch zufaellige Transformationen desselben Bildes.
# Das Modell sieht nie zweimal exakt dasselbe Bild.
# Augmentation ist NUR beim Training aktiv, nicht beim Testen.
#
# Hinweis fuer CIFAKE: Augmentation kann bei subtilen Pixel-Unterschieden
# schaden, weil die feinen KI-Artefakte dadurch veraendert werden.
# Wir testen es trotzdem, um den Effekt zu demonstrieren.

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),  # Bild zufaellig spiegeln
    layers.RandomRotation(0.1),       # Rotation bis +/-10 Grad
    layers.RandomZoom(0.1),           # Zoom bis +/-10 Prozent
    layers.RandomContrast(0.1),       # Kontrast leicht veraendern
], name="data_augmentation")

# Augmentierung visualisieren: dasselbe Bild, 8 verschiedene Versionen
plt.figure(figsize=(10, 4))
plt.suptitle("Data Augmentation: dasselbe Bild, 8x transformiert", fontsize=11)
for images, _ in train_ds.take(1):
    ein_bild = images[0]
    for i in range(8):
        plt.subplot(2, 4, i+1)
        augmented = data_augmentation(tf.expand_dims(ein_bild, 0))
        plt.imshow(augmented[0].numpy().astype("uint8"))
        plt.axis("off")
plt.tight_layout()
plt.show()


# In[14]:


# Dropout: setzt zufaellig X% der Neuronen auf 0 waehrend des Trainings
# -> Modell kann sich nicht auf einzelne Neuronen verlassen
# -> erzwingt robusteres Lernen
# -> reduziert Overfitting bei kleinen Datensaetzen
# Bei grossen Datensaetzen (wie hier 100k) kann Dropout die Accuracy senken

model_improved = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(32,  (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64,  (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
], name="model_improved")

model_improved.summary()

model_improved.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_imp = model_improved.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )]
)

imp_loss, imp_acc = model_improved.evaluate(test_ds, verbose=0)
print(f"Verbessertes Modell Test-Accuracy: {imp_acc:.2%}")


# In[15]:


plot_history(history_imp, "Modell B (Augmentation + Dropout) — Lernkurven")

# Direkter Vergleich beider Modelle
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Modellvergleich: Basismodell vs. Augmentation + Dropout", fontsize=12)

axes[0].plot(history_base.history["val_accuracy"], label="Basismodell",  color="steelblue")
axes[0].plot(history_imp.history["val_accuracy"],  label="Aug + Dropout", color="tomato")
axes[0].set_title("Validation Accuracy")
axes[0].set_xlabel("Epoche")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_base.history["val_loss"], label="Basismodell",  color="steelblue")
axes[1].plot(history_imp.history["val_loss"],  label="Aug + Dropout", color="tomato")
axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoche")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("=" * 42)
print(f"  Basismodell:          {base_acc:.2%}")
print(f"  Modell + Aug/Dropout: {imp_acc:.2%}")
diff = imp_acc - base_acc
print(f"  Differenz:           {diff:+.2%}")
print("=" * 42)
# Interpretation: wenn Basismodell besser ist, war Augmentation
# bei diesem Datensatz nicht hilfreich (subtile Artefakte wurden zerstoert)


# In[17]:


# Alle 20.000 Testbilder vorhersagen und auswerten
all_preds  = []
all_labels = []

for images, labels in test_ds:
    preds = model_base.predict(images, verbose=0)
    all_preds.extend(preds.flatten())
    all_labels.extend(labels.numpy().flatten())

all_preds    = np.array(all_preds)
all_labels   = np.array(all_labels).astype(int)
pred_classes = (all_preds > 0.5).astype(int)
# > 0.5 -> REAL (1), <= 0.5 -> FAKE (0)

# Konfusionsmatrix:
# Zeile = tatsaechliches Label
# Spalte = vorhergesagtes Label
# Diagonale = richtig klassifiziert
cm = confusion_matrix(all_labels, pred_classes)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=["FAKE", "REAL"],
    yticklabels=["FAKE", "REAL"],
    cmap="Blues"
)
plt.title("Konfusionsmatrix — Basismodell")
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsaechlich")
plt.tight_layout()
plt.show()

print(f"FAKE korrekt als FAKE erkannt: {cm[0,0]:5d} / {cm[0,0]+cm[0,1]}")
print(f"REAL korrekt als REAL erkannt: {cm[1,1]:5d} / {cm[1,0]+cm[1,1]}")
print(f"FAKE faelschlich als REAL:     {cm[0,1]:5d}")
print(f"REAL faelschlich als FAKE:     {cm[1,0]:5d}")

print("\nDetaillierter Bericht:")
print(classification_report(all_labels, pred_classes,
                             target_names=["FAKE", "REAL"]))


# In[18]:


# Bilder fuer Anzeige groesser laden (128x128 statt 32x32)
# Modell bekommt weiterhin 32x32 - nur die Anzeige ist groesser
raw_test = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_PATH, "test"),
    image_size=(128, 128),
    batch_size=10,
    seed=99,
    label_mode="binary",
    shuffle=True
)

for images_raw, labels_raw in raw_test.take(1):
    images_small = tf.image.resize(images_raw, (32, 32))
    preds_batch  = model_base.predict(tf.cast(images_small, tf.float32), verbose=0)

    fig, axes = plt.subplots(10, 2, figsize=(8, 22))
    fig.suptitle("Vorhersagen — Bild + Wahrscheinlichkeiten", fontsize=13)

    for i in range(10):
        prob_fake  = 1 - float(preds_batch[i][0])
        prob_real  =     float(preds_batch[i][0])
        true_label = class_names[int(labels_raw[i])]
        pred_label = "REAL" if prob_real > 0.5 else "FAKE"
        richtig    = pred_label == true_label

        # Bild
        axes[i, 0].imshow(images_raw[i].numpy().astype("uint8"))
        axes[i, 0].axis("off")
        axes[i, 0].set_title(
            f"Echt: {true_label}  |  Pred: {pred_label}",
            fontsize=9,
            color="green" if richtig else "red"
        )

        # Balkendiagramm
        balken_farben = []
        for lbl in ["FAKE", "REAL"]:
            if lbl == pred_label:
                balken_farben.append("green" if richtig else "red")
            else:
                balken_farben.append("lightgray")

        bars = axes[i, 1].barh(
            ["FAKE", "REAL"], [prob_fake, prob_real],
            color=balken_farben, height=0.5
        )
        for bar, prob in zip(bars, [prob_fake, prob_real]):
            axes[i, 1].text(
                min(bar.get_width() + 0.02, 0.95),
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.0%}", va="center", fontsize=9, fontweight="bold"
            )
        axes[i, 1].set_xlim(0, 1)
        axes[i, 1].axvline(0.5, color="black", linewidth=0.8, linestyle="--")
        axes[i, 1].set_xlabel("Wahrscheinlichkeit")

    plt.tight_layout()
    plt.show()


# In[19]:


# ── Bilder + Histogramm nebeneinander ────────────────────────────────────

raw_test = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_PATH, "test"),
    image_size=(128, 128),   # ← größer für bessere Anzeige!
    batch_size=10,
    seed=99,
    label_mode="binary",
    shuffle=True
)

for images_raw, labels_raw in raw_test.take(1):
    # Vorhersagen mit dem original-skalierten Input
    images_small = tf.image.resize(images_raw, (32, 32))
    preds_batch  = model_improved.predict(
                       tf.cast(images_small, tf.float32), verbose=0)

    n = 10
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 2.2))
    fig.suptitle("Vorhersagen mit Wahrscheinlichkeits-Histogramm",
                 fontsize=13, y=1.01)

    for i in range(n):
        prob_fake = 1 - float(preds_batch[i][0])  # FAKE=0 im Dataset
        prob_real =     float(preds_batch[i][0])  # REAL=1 im Dataset
        true_label    = class_names[int(labels_raw[i])]
        pred_label    = "REAL" if prob_real > 0.5 else "FAKE"
        richtig       = pred_label == true_label

        # ── Linke Spalte: Bild ────────────────────────────────────────────
        ax_img = axes[i, 0]
        ax_img.imshow(images_raw[i].numpy().astype("uint8"))
        ax_img.axis("off")
        ax_img.set_title(
            f"Echt: {true_label}",
            fontsize=9,
            color="green" if richtig else "red"
        )

        # ── Rechte Spalte: Histogramm ─────────────────────────────────────
        ax_bar = axes[i, 1]
        balken_farben = []
        for label in ["FAKE", "REAL"]:
            if label == pred_label and richtig:
                balken_farben.append("green")
            elif label == pred_label and not richtig:
                balken_farben.append("red")
            else:
                balken_farben.append("lightgray")

        bars = ax_bar.barh(
            ["FAKE", "REAL"],
            [prob_fake, prob_real],
            color=balken_farben,
            height=0.5
        )

        # Prozentwerte in die Balken schreiben
        for bar, prob in zip(bars, [prob_fake, prob_real]):
            w = bar.get_width()
            ax_bar.text(
                min(w + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                f"{prob:.0%}",
                va="center", fontsize=9, fontweight="bold"
            )

        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Wahrscheinlichkeit")
        ax_bar.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
        ax_bar.set_title(
            f"→ {pred_label} ({'✓' if richtig else '✗'})",
            fontsize=9,
            color="green" if richtig else "red"
        )

    plt.tight_layout()
    plt.show()


# In[20]:


def predict_cifake(image_source, is_url=False):
    """
    Laedt ein externes Bild (URL oder Dateipfad) und zeigt
    die Vorhersagen beider Modelle mit Balkendiagramm.

    Parameter:
        image_source: URL-String oder lokaler Dateipfad
        is_url:       True fuer URL, False fuer lokalen Pfad
    """

    # Bild laden und auf RGB sicherstellen
    # .convert("RGB") verhindert Fehler bei PNG mit Transparenz (4 Kanaele)
    if is_url:
        response = requests.get(image_source)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")

    # Zwei Versionen: gross fuer Anzeige, klein fuer Modell
    img_display = img.resize((128, 128))
    img_model   = img.resize((32, 32))

    # Vorverarbeitung: NumPy-Array, Batch-Dimension hinzufuegen
    # expand_dims: (32,32,3) -> (1,32,32,3)
    # Das Modell erwartet immer einen Batch, auch bei einem einzelnen Bild
    img_array = np.array(img_model).astype("float32")
    img_batch = np.expand_dims(img_array, axis=0)

    # Vorhersage: sigmoid gibt einen Wert zwischen 0 und 1
    # nahe 0 -> FAKE, nahe 1 -> REAL
    pred_base = float(model_base.predict(img_batch, verbose=0)[0][0])
    pred_imp  = float(model_improved.predict(img_batch, verbose=0)[0][0])

    modelle = {
        "Basismodell":         {"REAL": pred_base, "FAKE": 1 - pred_base},
        "Modell Aug/Dropout":  {"REAL": pred_imp,  "FAKE": 1 - pred_imp},
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("CIFAKE — Externe Bildvorhersage", fontsize=13)

    axes[0].imshow(img_display)
    axes[0].axis("off")
    axes[0].set_title("Eingabebild\n(Anzeige: 128x128)", fontsize=9)

    for ax, (modell_name, probs) in zip(axes[1:], modelle.items()):
        pred_label = "REAL" if probs["REAL"] > 0.5 else "FAKE"
        sicherheit = max(probs["REAL"], probs["FAKE"])

        farben = []
        for lbl in ["FAKE", "REAL"]:
            if lbl == pred_label:
                farben.append("green" if pred_label == "REAL" else "tomato")
            else:
                farben.append("lightgray")

        bars = ax.barh(
            ["FAKE", "REAL"],
            [probs["FAKE"], probs["REAL"]],
            color=farben, height=0.5
        )
        for bar, (lbl, prob) in zip(bars, [("FAKE", probs["FAKE"]),
                                            ("REAL", probs["REAL"])]):
            ax.text(
                max(bar.get_width() - 0.05, 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}",
                va="center", ha="right",
                fontsize=10, fontweight="bold", color="white"
            )

        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Wahrscheinlichkeit")
        ax.set_title(
            f"{modell_name}\n-> {pred_label} ({sicherheit:.1%} sicher)",
            fontsize=9,
            color="green" if pred_label == "REAL" else "tomato"
        )

    plt.tight_layout()
    plt.show()

    print(f"Basismodell:        REAL={pred_base:.1%}  FAKE={1-pred_base:.1%}  -> {('REAL' if pred_base > 0.5 else 'FAKE')}")
    print(f"Aug/Dropout-Modell: REAL={pred_imp:.1%}  FAKE={1-pred_imp:.1%}  -> {('REAL' if pred_imp > 0.5 else 'FAKE')}")
    beide_gleich = (pred_base > 0.5) == (pred_imp > 0.5)
    print(f"Beide einig: {'Ja' if beide_gleich else 'Nein — Modelle widersprechen sich'}")


# In[1]:


# Test 1: Bild von Lexica.art (Stable Diffusion generiert) -> sollte FAKE sein
url_sd = "https://image.lexica.art/full_webp/0864114e-c958-4401-a1bb-d2dfdba31534"
predict_cifake(url_sd, is_url=True)

# Test 2: Bild direkt aus dem FAKE-Testordner -> garantiert FAKE (Stable Diffusion 1.4)
fake_datei = os.path.join(DATA_PATH, "test", "FAKE",
                           os.listdir(os.path.join(DATA_PATH, "test", "FAKE"))[0])
predict_cifake(fake_datei, is_url=False)

# Test 3: Bild direkt aus dem REAL-Testordner -> garantiert REAL
real_datei = os.path.join(DATA_PATH, "test", "REAL",
                           os.listdir(os.path.join(DATA_PATH, "test", "REAL"))[0])
predict_cifake(real_datei, is_url=False)

# Test 4: Eigenes lokales Bild
# predict_cifake("mein_bild.jpg", is_url=False)


# In[3]:


# KI-generiertes Bild testen (z.B. von einem AI-Image-Generator)
# url_fake = "https://..."
# predict_cifake(url_fake, is_url=True)

#Lokales Bild testen
predict_cifake("aiCat.webp", is_url=False)


# In[ ]:




