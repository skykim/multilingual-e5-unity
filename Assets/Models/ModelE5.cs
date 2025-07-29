using UnityEngine;
using Microsoft.ML.Tokenizers;
using Unity.InferenceEngine;
using FF = Unity.InferenceEngine.Functional;
using System.IO;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

public class ModelE5 : MonoBehaviour
{
    public enum E5ModelType { E5_SMALL, E5_BASE, E5_LARGE }

    [Header("Model Selection")]
    public E5ModelType selectedModel = E5ModelType.E5_SMALL;

    [Header("Test (prefix: 'query: ' or 'passage: ')")]
    public string Query = "";
    public List<string> Passages = new List<string>();

    private SentencePieceTokenizer _tokenizer;
    private Worker _engine;
    private Worker _dotScore;
    private const BackendType BACKEND = BackendType.GPUCompute;

    private int _features;
    private E5ModelType _activeModelType;

    void Start()
    {
        string modelAssetPath = "";
        _activeModelType = selectedModel;

        switch (_activeModelType)
        {
            case E5ModelType.E5_LARGE:
                _features = 1024;
                modelAssetPath = Path.Combine(Application.streamingAssetsPath, "e5-large_fp16.sentis");
                break;
            case E5ModelType.E5_BASE:
                _features = 768;
                modelAssetPath = Path.Combine(Application.streamingAssetsPath, "e5-base_fp16.sentis");
                break;
            case E5ModelType.E5_SMALL:
                _features = 384;
                modelAssetPath = Path.Combine(Application.streamingAssetsPath, "e5-small_fp16.sentis");
                break;
            default:
                Debug.LogError("Unsupported model type selected.");
                return;
        }

        var tokenizerModelPath = Path.Combine(Application.streamingAssetsPath, "sentencepiece.bpe.model");
        using (Stream tokenizerModelStream = new FileStream(tokenizerModelPath, FileMode.Open, FileAccess.Read))
        {
            _tokenizer = SentencePieceTokenizer.Create(
                tokenizerModelStream
            );
        }

        Model baseModel = ModelLoader.Load(modelAssetPath);
        FunctionalGraph modelWithPoolingGraph = new FunctionalGraph();

        FunctionalTensor tokenEmbeddings;
        FunctionalTensor attentionMaskInput;

        if (_activeModelType == E5ModelType.E5_BASE || _activeModelType == E5ModelType.E5_LARGE)
        {
            FunctionalTensor[] inputs = new FunctionalTensor[2];
            inputs[0] = modelWithPoolingGraph.AddInput(baseModel, 0);
            inputs[1] = modelWithPoolingGraph.AddInput(baseModel, 1);
            attentionMaskInput = inputs[1];
            tokenEmbeddings = FF.Forward(baseModel, inputs)[0];
        }
        else
        {
            FunctionalTensor[] inputs = new FunctionalTensor[3];
            inputs[0] = modelWithPoolingGraph.AddInput(baseModel, 0);
            inputs[1] = modelWithPoolingGraph.AddInput(baseModel, 1);
            inputs[2] = modelWithPoolingGraph.AddInput(baseModel, 2);
            attentionMaskInput = inputs[1];
            tokenEmbeddings = FF.Forward(baseModel, inputs)[0];
        }

        FunctionalTensor meanPooling = MeanPooling(tokenEmbeddings, attentionMaskInput);
        Model finalModel = modelWithPoolingGraph.Compile(meanPooling);

        FunctionalGraph dotScoreGraph = new FunctionalGraph();
        FunctionalTensor x = dotScoreGraph.AddInput<float>(new TensorShape(1, _features));
        FunctionalTensor y = dotScoreGraph.AddInput<float>(new TensorShape(1, _features));
        FunctionalTensor reduce = FF.ReduceSum(x * y, 1);
        Model dotScoreModel = dotScoreGraph.Compile(reduce);

        _engine = new Worker(finalModel, BACKEND);
        _dotScore = new Worker(dotScoreModel, BACKEND);

        StartCoroutine(RunSimilarityTest());
    }

    IEnumerator RunSimilarityTest()
    {
        yield return null;

        if (string.IsNullOrEmpty(Query) || Passages.Count == 0)
        {
            Debug.LogWarning("Query or Passages list is empty. No similarities to calculate.");
            yield break;
        }

        foreach (var passage in Passages)
        {
            float similarity = CalculateSimilarity(Query, passage);
            Debug.Log($"Similarity between Query and '{passage}': {similarity}");
        }
    }

    FunctionalTensor MeanPooling(FunctionalTensor tokenEmbeddings, FunctionalTensor attentionMask)
    {
        var mask = attentionMask.Unsqueeze(-1).BroadcastTo(new[] { _features });
        var A = FF.ReduceSum(tokenEmbeddings * mask, 1, false);
        var B = A / (FF.ReduceSum(mask, 1, false) + 1e-9f);
        var C = FF.Sqrt(FF.ReduceSum(FF.Square(B), 1, true));
        return B / C;
    }

    public float CalculateSimilarity(string text1, string text2)
    {
        int[] rawIds1 = _tokenizer.EncodeToIds(text1, false, false).ToArray();
        int[] rawIds2 = _tokenizer.EncodeToIds(text2, false, false).ToArray();

        // Reason: The output of Microsoft.ML.Tokenizers has one less value than the Python example.
        var processedIds1 = rawIds1.Select(id => id + 1).ToList();
        var processedIds2 = rawIds2.Select(id => id + 1).ToList();

        processedIds1.Insert(0, 0);
        processedIds1.Add(2);
        processedIds2.Insert(0, 0);
        processedIds2.Add(2);

        int originalLength1 = processedIds1.Count;
        int originalLength2 = processedIds2.Count;
        int maxLength = Mathf.Max(originalLength1, originalLength2);

        if (originalLength1 < maxLength)
        {
            processedIds1.AddRange(Enumerable.Repeat(1, maxLength - originalLength1));
        }
        else if (originalLength2 < maxLength)
        {
            processedIds2.AddRange(Enumerable.Repeat(1, maxLength - originalLength2));
        }

        var attentionMask1 = Enumerable.Repeat(1, originalLength1).Concat(Enumerable.Repeat(0, maxLength - originalLength1)).ToList();
        var attentionMask2 = Enumerable.Repeat(1, originalLength2).Concat(Enumerable.Repeat(0, maxLength - originalLength2)).ToList();

        using Tensor<float> embedding1 = GetEmbedding(processedIds1.ToArray(), attentionMask1.ToArray());
        using Tensor<float> embedding2 = GetEmbedding(processedIds2.ToArray(), attentionMask2.ToArray());

        float score = GetDotScore(embedding1, embedding2);
        return score;
    }


    private Tensor<float> GetEmbedding(int[] processedInputIds, int[] attentionMask)
    {
        int N = processedInputIds.Length;
        var shape = new TensorShape(1, N);

        using var inputIdsTensor = new Tensor<int>(shape, processedInputIds);
        using var attentionMaskTensor = new Tensor<int>(shape, attentionMask);

        _engine.SetInput("input_ids", inputIdsTensor);
        _engine.SetInput("attention_mask", attentionMaskTensor);

        if (_activeModelType == E5ModelType.E5_SMALL)
        {
            using var tokenTypeIdsTensor = new Tensor<int>(shape, Enumerable.Repeat(0, N).ToArray());
            _engine.SetInput("token_type_ids", tokenTypeIdsTensor);
        }

        _engine.Schedule();
        var output = _engine.PeekOutput() as Tensor<float>;

        return output.ReadbackAndClone();
    }

    private float GetDotScore(Tensor<float> A, Tensor<float> B)
    {
        _dotScore.SetInput("input_0", A);
        _dotScore.SetInput("input_1", B);
        _dotScore.Schedule();

        var output = _dotScore.PeekOutput() as Tensor<float>;
        if (output != null)
        {
            using (var cpuOutput = output.ReadbackAndClone())
            {
                return cpuOutput[0];
            }
        }
        else
        {
            Debug.LogError("Failed to compute dot score. Output is null.");
            return 0;
        }
    }

    private void OnDestroy()
    {
        _engine?.Dispose();
        _dotScore?.Dispose();
    }
}