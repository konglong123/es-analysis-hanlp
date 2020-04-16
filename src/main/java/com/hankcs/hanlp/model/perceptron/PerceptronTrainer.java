/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-10-26 下午5:51</create-date>
 *
 * <copyright file="PerceptronTrainer.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.perceptron;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.model.perceptron.common.FrequencyMap;
import com.hankcs.hanlp.model.perceptron.feature.ImmutableFeatureMap;
import com.hankcs.hanlp.model.perceptron.feature.MutableFeatureMap;
import com.hankcs.hanlp.model.perceptron.instance.Instance;
import com.hankcs.hanlp.model.perceptron.model.AveragedPerceptron;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.model.perceptron.model.StructuredPerceptron;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;
import com.hankcs.hanlp.model.perceptron.utility.IOUtility;
import com.hankcs.hanlp.model.perceptron.instance.InstanceHandler;
import com.hankcs.hanlp.model.perceptron.utility.Utility;
import com.hankcs.hanlp.classification.utilities.io.ConsoleLogger;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.document.sentence.Sentence;

import java.io.*;
import java.util.*;

import static java.lang.System.err;
import static java.lang.System.out;

/**
 * 感知机训练基类
 *
 * @author hankcs
 */
public abstract class PerceptronTrainer extends InstanceConsumer
{

    /**
     * 训练结果
     */
    public static class Result
    {
        /**
         * 模型
         */
        LinearModel model;
        /**
         * 精确率(Precision), 召回率(Recall)和F1-Measure<br>
         * 中文参考：https://blog.argcv.com/articles/1036.c
         */
        double prf[];

        //训练过程
        StringBuilder message;

        List<String> progress;

        public Result(LinearModel model, double[] prf,StringBuilder message,List<String> progress)
        {
            this.model = model;
            this.prf = prf;
            this.message=message;
            this.progress=progress;
        }

        /**
         * 获取准确率
         *
         * @return
         */
        public double getAccuracy()
        {
            if (prf.length == 3)
            {
                return prf[2];
            }
            return prf[0];
        }

        /**
         * 获取模型
         *
         * @return
         */
        public LinearModel getModel()
        {
            return model;
        }

        public StringBuilder getMessage(){
            return message;
        }

        public List<String> getProgress(){return progress;}


    }

    /**
     * 创建标注集
     *
     * @return
     */
    protected abstract TagSet createTagSet();

    /**
     * 训练
     *
     * @param trainingFile  训练集
     * @param developFile   开发集
     * @param modelFile     模型保存路径
     * @param compressRatio 压缩比
     * @param maxIteration  最大迭代次数
     * @param threadNum     线程数
     * @return 一个包含模型和精度的结构
     * @throws IOException
     */
    public Result train(String trainingFile, String developFile,
                        String modelFile, final double compressRatio,
                        final int maxIteration, final int threadNum) throws IOException
    {
        if (developFile == null)
        {
            developFile = trainingFile;
        }

        StringBuilder message=new StringBuilder();
        List<String> accuracyList=new ArrayList<>();

        // 加载训练语料
        TagSet tagSet = createTagSet();
        MutableFeatureMap mutableFeatureMap = new MutableFeatureMap(tagSet);
        message.append("--开始加载训练集...");
        Instance[] instances = loadTrainInstances(trainingFile, mutableFeatureMap);
        tagSet.lock();
        message.append("--加载完毕，实例一共"+ instances.length+"句，特征总数"+mutableFeatureMap.size() * tagSet.size());

        // 开始训练
        ImmutableFeatureMap immutableFeatureMap = new ImmutableFeatureMap(mutableFeatureMap.featureIdMap, tagSet);
        mutableFeatureMap = null;
        double[] accuracy = null;

        if (threadNum == 1)
        {
            AveragedPerceptron model;
            model = new AveragedPerceptron(immutableFeatureMap);
            final double[] total = new double[model.parameter.length];
            final int[] timestamp = new int[model.parameter.length];
            int current = 0;
            for (int iter = 1; iter <= maxIteration; iter++)
            {
                Utility.shuffleArray(instances);
                for (Instance instance : instances)
                {
                    ++current;
                    int[] guessLabel = new int[instance.length()];
                    model.viterbiDecode(instance, guessLabel);
                    for (int i = 0; i < instance.length(); i++)
                    {
                        int[] featureVector = instance.getFeatureAt(i);
                        int[] goldFeature = new int[featureVector.length];
                        int[] predFeature = new int[featureVector.length];
                        for (int j = 0; j < featureVector.length - 1; j++)
                        {
                            goldFeature[j] = featureVector[j] * tagSet.size() + instance.tagArray[i];
                            predFeature[j] = featureVector[j] * tagSet.size() + guessLabel[i];
                        }
                        goldFeature[featureVector.length - 1] = (i == 0 ? tagSet.bosId() : instance.tagArray[i - 1]) * tagSet.size() + instance.tagArray[i];
                        predFeature[featureVector.length - 1] = (i == 0 ? tagSet.bosId() : guessLabel[i - 1]) * tagSet.size() + guessLabel[i];
                        model.update(goldFeature, predFeature, total, timestamp, current);
                    }
                }

                // 在开发集上校验
                accuracy = trainingFile.equals(developFile) ? IOUtility.evaluate(instances, model) : evaluate(developFile, model);
                printAccuracy(accuracy,accuracyList);
            }
            // 平均
            model.average(total, timestamp, current);
            accuracy = trainingFile.equals(developFile) ? IOUtility.evaluate(instances, model) : evaluate(developFile, model);
            message.append("--AP - ");
            printAccuracy(accuracy,accuracyList);
            message.append("--以压缩比 "+(compressRatio+"").substring(0,8)+" 保存模型到 ... "+ modelFile);
            model.save(modelFile, immutableFeatureMap.featureIdMap.entrySet(), compressRatio);
            message.append("--保存完毕--");
            if (compressRatio == 0) return new Result(model, accuracy,message,accuracyList);
        }
        else
        {
            // 多线程用Structure Perceptron
            StructuredPerceptron[] models = new StructuredPerceptron[threadNum];
            for (int i = 0; i < models.length; i++)
            {
                models[i] = new StructuredPerceptron(immutableFeatureMap);
            }

            TrainingWorker[] workers = new TrainingWorker[threadNum];
            int job = instances.length / threadNum;
            for (int iter = 1; iter <= maxIteration; iter++)
            {
                Utility.shuffleArray(instances);
                try
                {
                    for (int i = 0; i < workers.length; i++)
                    {
                        workers[i] = new TrainingWorker(instances, i * job,
                                                        i == workers.length - 1 ? instances.length : (i + 1) * job,
                                                        models[i]);
                        workers[i].start();
                    }
                    for (TrainingWorker worker : workers)
                    {
                        worker.join();
                    }
                    for (int j = 0; j < models[0].parameter.length; j++)
                    {
                        for (int i = 1; i < models.length; i++)
                        {
                            models[0].parameter[j] += models[i].parameter[j];
                        }
                        models[0].parameter[j] /= threadNum;
                    }
                    accuracy = trainingFile.equals(developFile) ? IOUtility.evaluate(instances, models[0]) : evaluate(developFile, models[0]);
                    printAccuracy(accuracy,accuracyList);
                }
                catch (InterruptedException e)
                {
                    message.append("--线程同步异常，训练失败--");
                    e.printStackTrace();
                    return null;
                }
            }
            message.append("--以压缩比 "+(compressRatio+"").substring(0,8)+" 保存模型到 ... "+modelFile);
            models[0].save(modelFile, immutableFeatureMap.featureIdMap.entrySet(), compressRatio, HanLP.Config.DEBUG);
            message.append("--保存完毕--");
            if (compressRatio == 0) return new Result(models[0], accuracy,message,accuracyList);
        }

        LinearModel model = new LinearModel(modelFile);
        if (compressRatio > 0)
        {
            accuracy = evaluate(developFile, model);
            message.append("--compressed model - "+ (compressRatio+"").substring(0,8));
            printAccuracy(accuracy,accuracyList);
        }

        return new Result(model, accuracy,message,accuracyList);
    }

    private void printAccuracy(double[] accuracy,List<String> list)
    {
        if (accuracy.length == 3)
        {
            String p=(accuracy[0]+"").substring(0,8);
            String r=(accuracy[1]+"").substring(0,8);
            String f=(accuracy[2]+"").substring(0,8);
            list.add("P:"+p+"R:"+r+ "F:"+f);
        }
        else
        {
            String p=(accuracy[0]+"").substring(0,8);
            list.add("P:"+p);
        }
    }

    private static class TrainingWorker extends Thread
    {
        private Instance[] instances;
        private int start;
        private int end;
        private StructuredPerceptron model;

        public TrainingWorker(Instance[] instances, int start, int end, StructuredPerceptron model)
        {
            this.instances = instances;
            this.start = start;
            this.end = end;
            this.model = model;
        }

        @Override
        public void run()
        {
            for (int s = start; s < end; ++s)
            {
                Instance instance = instances[s];
                model.update(instance);
            }
//            out.printf("Finished [%d,%d)\n", start, end);
        }
    }

    protected Instance[] loadTrainInstances(String trainingFile, final MutableFeatureMap mutableFeatureMap) throws IOException
    {
        final List<Instance> instanceList = new LinkedList<Instance>();
        IOUtility.loadInstance(trainingFile, new InstanceHandler()
        {
            @Override
            public boolean process(Sentence sentence)
            {
                Utility.normalize(sentence);
                instanceList.add(PerceptronTrainer.this.createInstance(sentence, mutableFeatureMap));
                return false;
            }
        });
        Instance[] instances = new Instance[instanceList.size()];
        instanceList.toArray(instances);
        return instances;
    }


    private static DoubleArrayTrie<Integer> loadDictionary(String trainingFile, String dictionaryFile) throws IOException
    {
        FrequencyMap dictionaryMap = new FrequencyMap();
        if (dictionaryFile == null)
        {
            out.printf("从训练文件%s中统计词库...\n", trainingFile);
            loadWordFromFile(trainingFile, dictionaryMap, true);
        }
        else
        {
            out.printf("从外部词典%s中加载词库...\n", trainingFile);
            loadWordFromFile(dictionaryFile, dictionaryMap, false);
        }
        DoubleArrayTrie<Integer> dat = new DoubleArrayTrie<Integer>();
        dat.build(dictionaryMap);
        out.printf("加载完毕，词库总词数：%d，总词频：%d\n", dictionaryMap.size(), dictionaryMap.totalFrequency);

        return dat;
    }

    public Result train(String trainingFile, String modelFile) throws IOException
    {
        return train(trainingFile, trainingFile, modelFile);
    }

    public Result train(String trainingFile, String developFile, String modelFile) throws IOException
    {
        return train(trainingFile, developFile, modelFile, 0.1, 50, Runtime.getRuntime().availableProcessors());
    }

    private static void loadWordFromFile(String path, FrequencyMap storage, boolean segmented) throws IOException
    {
        BufferedReader br = IOUtility.newBufferedReader(path);
        String line;
        while ((line = br.readLine()) != null)
        {
            if (segmented)
            {
                for (String word : IOUtility.readLineToArray(line))
                {
                    storage.add(word);
                }
            }
            else
            {
                line = line.trim();
                if (line.length() != 0)
                {
                    storage.add(line);
                }
            }
        }
        br.close();
    }
}
