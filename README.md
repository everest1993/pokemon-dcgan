# Pokemon DCGAN

Progetto di **Generative Adversarial Network (DCGAN)** per generare immagini di Pokemon a partire dal dataset Kaggle `vishalsubbiah/pokemon-images-and-types`.

## Cosa fa il progetto

Il training mette in competizione due reti:

- **Generatore**: parte da rumore casuale e crea immagini RGB `120x120`.
- **Discriminatore**: prova a distinguere immagini reali del dataset da immagini generate.

Durante le epoche, il generatore impara a produrre immagini sempre piu realistiche.

## Struttura file

- `config.py`: sceglie e stampa il device TensorFlow (`GPU` o `CPU`).
- `dcgan.py`: definisce architetture di generatore/discriminatore e la classe `DCGAN` con `train_step` personalizzato.
- `main.py`: scarica dataset, carica e normalizza immagini, compila il modello e avvia training.

## Come funziona

1. `main.py` scarica il dataset con `kagglehub`.
2. Le immagini vengono lette con OpenCV, convertite in RGB e normalizzate in intervallo `[-1, 1]`.
3. In `dcgan.py`:
   - il generatore usa layer `Dense + Conv2DTranspose` per fare upsampling fino a `120x120x3`;
   - il discriminatore usa `Conv2D` per fare downsampling e classificare reale/falso.
4. Loss binaria (`BinaryCrossentropy`). Gli ottimizzatori sono Adam con learning rate diverso tra G e D.
5. Ogni 20 epoche il callback `ShowImg` mostra un campione di immagini generate.

## Installazione

Da `GAN/pokemon-gan`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Opzionale (Mac Apple Silicon con Metal):

```bash
pip install tensorflow-metal
```

Nota: `tensorflow-metal` non è disponibile su Python `3.13` (supporta `3.11/3.12`).

## Utilizzo

Da `GAN/pokemon-gan`:

```bash
python3 main.py
```

Alla prima esecuzione il dataset viene scaricato; poi parte il training (`epochs=200`) con visualizzazione periodica delle immagini generate.

## Note

- In `main.py` e gia presente `get_device(use_apple_gpu=True)` per tentare l'uso della GPU Apple (Metal).
- Modifica parametri e numero di epoche per personalizzare il training
