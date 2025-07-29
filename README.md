# Multilingual-E5 for Unity

[![Unity Version](https://img.shields.io/badge/Unity-6000.0.50f1+-black.svg?style=for-the-badge&logo=unity)](https://unity.com/)
[![Inference Engine](https://img.shields.io/badge/Inference-2.2.1-blue.svg?style=for-the-badge)](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.2/manual/index.html)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/collections/intfloat/multilingual-e5-text-embeddings-67b2b8bb9bff40dec9fb3534)

This project enables the use of [Multilingual E5](https://arxiv.org/abs/2402.05672) models within the Unity AI Inference Engine (formerly Sentis).

Multilingual E5, proposed by Liang et al. in 2024, is a state-of-the-art multilingual embedding model that supports over 100 languages. This project allows you to leverage powerful multilingual text embedding and similarity search capabilities directly in your Unity projects.

---

## ✨ Key Features

- **Multilingual Support**: Supports text embeddings for over 100 languages.
- **State-of-the-Art Models**: Based on the E5 models (Small, Base, Large) from `intfloat`.
- **Optimized for Unity**: Efficiently runs ONNX models using Unity's Inference Engine.
- **Versatile Applications**: Can be used for various NLP tasks, such as sentence similarity, information retrieval, and more.

## ⚙️ Requirements

- **Unity**: `6000.0.50f1`
- **Inference Engine**: `2.2.1`

## 🧠 Models (ONNX)

You can download the Multilingual E5 models from the `intfloat` repository on Hugging Face.

| Model Type | Hugging Face Link |
| :--- | :--- |
| **Small** | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) |
| **Base** | [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) |
| **Large** | [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) |

**Important**: To use the models, you need both the `model.onnx` file and the `sentencepiece.bpe.model` file from the `onnx` directory of each model repository. You can optimize the models to `fp16` or `int8` using the quantization features in Unity's Inference Engine.

## 🚀 Getting Started

### 1. Project Setup

- Clone or download this repository.
- Unzip the provided [StreamingAssets.zip](https://drive.google.com/file/d/1j_YW7SJTRZM0DwN8nugjjYO-9XE_h2OK/view?usp=sharing) file and place its contents into the `/Assets/StreamingAssets` directory in your project.

### 2. Run the Demo Scene

- Open the `/Assets/Scenes/E5Scene.unity` scene in the Unity Editor.
- Run the scene to see the multilingual embedding tests in action.

## 💡 How to Use

The E5 models require a prefix for each input text to specify its purpose.

- **`query:`**: Use for the sentence you want to search for or compare. (e.g., `query: What is the weather like today?`)
- **`passage:`**: Use for the document or paragraph that is being searched. (e.g., `passage: Seoul is experiencing clear weather today.`)

For tasks other than retrieval, such as simple similarity comparison, you can use the `query:` prefix for all inputs. This rule applies to **non-English texts as well**.

## 🧪 Test Cases

### Task 1: Information Retrieval (Korean)

Find the most relevant `passage` for a given `query`.


**Query:**

```
query: 대한민국의 국민 총소득
```

**Results:**
```
(prob: 0.8888728) passage: 현재는, 2024년 국내총생산(GDP) 기준 세계 12위권의 경제 규모를 가지고 있으며, 2024년 1인당 국민 총소득(GNI)은 명목 3만 6,624달러이다.
(prob: 0.8283274) passage: 대한민국은 20세기 후반 이후 급격한 경제 성장을 이루었으나, 그 과정에서 1990년대 말 외환 위기 등의 부침이 있기도 했다.
(prob: 0.8263298) passage: 대한민국은 1948년 5월 10일 총선거를 통해 제헌국회를 구성하였고, 1948년 8월 15일 대한민국 정부를 수립하였다.
(prob: 0.8158849) passage: 대한민국의 국기는 대한민국 국기법에 따라 태극기이며, 국가는 관습상 애국가, 국화는 관습상 무궁화이다.
...
```

### Task 2: Sentence Similarity (Japanese)

Compare the similarity between two `query` sentences.

**Query:**
```
query: カメラ2番に切り替えてください
```

**Results:**
```
(prob: 0.986789) query: 2番カメラに切り替え
(prob: 0.9446367) query: 1番カメラに切り替え
(prob: 0.8789923) query: 雰囲気の切り替え
(prob: 0.8641801) query: ユーザー検知モードに切り替え
(prob: 0.8124454) query: 工場の巡回勤務を開始
```
