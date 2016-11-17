/**
 * Created by Hao Liu (hliu32@uncc.edu) on 9/27/16.
 */

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;


public class Search extends Configured implements Tool {

    private static final Logger LOG = Logger .getLogger( Search.class);

    public static void main( String[] args) throws  Exception {


        BufferedReader buffer = new BufferedReader(new InputStreamReader(System.in));

        // get query
        System.out.println("input query terms:");
        String text = buffer.readLine();

        // put query as the last argument.
        String[] newArgs = new String[args.length + 1];
        System.arraycopy(args, 0, newArgs, 0, args.length);
        newArgs[args.length] = text;

        int res  = ToolRunner.run(new Search(), newArgs);
        System .exit(res);
    }

    public int run( String[] args) throws  Exception {

        Configuration conf = getConf();

        // job1 for term frequency
        Job job1  = Job.getInstance(conf, " TermFrequency ");
        job1.setJarByClass(this.getClass());

        FileInputFormat.addInputPaths(job1, args[0]);
        FileOutputFormat.setOutputPath(job1, new Path(args[1]));
        job1.setMapperClass(Search.TFMap.class);
        job1.setReducerClass(Search.TFReduce.class);
        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(IntWritable.class);

        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(DoubleWritable.class);


        job1.waitForCompletion(true);

        // job2 for TFIDF
        conf.set("numDocs", args[3]);
        Job job2 = Job.getInstance(conf, "TFIDF");


        job2.setJarByClass(this.getClass());

        FileInputFormat.addInputPaths(job2, args[1]);
        FileOutputFormat.setOutputPath(job2, new Path(args[2]));

        job2.setMapperClass(Search.TFIDFMap.class);
        job2.setReducerClass(Search.TFIDFReduce.class);

        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);

        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(DoubleWritable.class);

        job2.waitForCompletion(true);

        //set query in configureation
        conf.set("query", args[args.length - 1]);
        // job3 for Search
        Job job3 = Job.getInstance(conf, "Search");

        job3.setJarByClass(this.getClass());

        FileInputFormat.addInputPaths(job3, args[2]);
        FileOutputFormat.setOutputPath(job3, new Path(args[4]));

        job3.setSortComparatorClass(LongWritable.DecreasingComparator.class);

        job3.setMapperClass(Search.SearchMap.class);
        job3.setReducerClass(Search.SearchReduce.class);

        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(DoubleWritable.class);

        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(DoubleWritable.class);

        return job3.waitForCompletion(true)  ? 0 : 1;
    }

    public static class TFMap extends Mapper<LongWritable,  Text,  Text,  IntWritable> {
        private final static IntWritable one  = new IntWritable( 1);
        private Text word  = new Text();

        private static final Pattern WORD_BOUNDARY = Pattern .compile("\\s*\\b\\s*");

        public void map( LongWritable offset,  Text lineText,  Context context)
                throws  IOException,  InterruptedException {
            String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
            String line  = lineText.toString();
            Text currentWord  = new Text();

            for ( String word  : WORD_BOUNDARY .split(line)) {
                if (word.isEmpty()) {
                    continue;
                }
                currentWord  = new Text(word + "#####" + fileName);
                context.write(currentWord,one);
            }
        }
    }

    public static class TFReduce extends Reducer<Text, IntWritable, Text, DoubleWritable> {
        //@Override
        public void reduce( Text word,  Iterable<IntWritable > counts,  Context context)
                throws IOException,  InterruptedException {
            int sum  = 0;
            for ( IntWritable count  : counts) {
                sum  += count.get();
            }
            double tmp = 1 + Math.log10(sum);
            context.write(word,  new DoubleWritable(tmp));
        }
    }

    public static class TFIDFMap extends Mapper<LongWritable, Text, Text, Text> {

        public void map(LongWritable offset, Text lineText, Context context)
                throws  IOException,  InterruptedException {


            String line  = lineText.toString();
            // reverse the line
            String newline = new StringBuffer(line).reverse().toString();
            // split the line, fields[1] is the word#####file, fields[0] is the term frequency
            String[] fields = newline.split("\t", 2);
            String tmp1 = new StringBuilder(fields[1]).reverse().toString();
            String[] parts = tmp1.split("#####", 2);
            // currentWord is word, currentVal is file=tf
            Text currentWord  = new Text(parts[0]);
            Text currentVal = new Text(parts[1] + "=" + new StringBuilder(fields[0]).reverse().toString());

            context.write(currentWord, currentVal);

        }

    }

    public static class TFIDFReduce extends Reducer<Text, Text, Text, DoubleWritable> {
        //@Override
        public void reduce( Text word,  Iterable<Text> values,  Context context)
                throws IOException,  InterruptedException {

            Configuration conf = context.getConfiguration();
            String numDocs = conf.get("numDocs");
            int sum  = Integer.valueOf(numDocs);
            HashMap<String, Double> termFrequencyMap = new HashMap<String, Double>();

            for (Text val : values) {
                String[] fields = val.toString().split("=");
                if(fields == null || fields.length != 2) {
                    continue;
                }
                termFrequencyMap.put(fields[0], Double.valueOf(fields[1]));

            }
            int size = termFrequencyMap.size();

            if(size == 0) {
                return;
            }
            double idf = Math.log10(1 + sum / size);

            for (String key : termFrequencyMap.keySet()) {
                Text currentKey = new Text(word.toString() + "@" + key);
                double tf = termFrequencyMap.get(key);
                double tmp = tf * idf;
                context.write(currentKey,  new DoubleWritable(tmp));
            }
        }
    }

    public static class SearchMap extends Mapper<LongWritable, Text, Text, DoubleWritable> {

        private static final Pattern WORD_BOUNDARY = Pattern .compile("\\s*\\b\\s*");

        public void map(LongWritable offset, Text lineText, Context context)
                throws  IOException,  InterruptedException {
            Configuration conf = context.getConfiguration();

            String line  = lineText.toString();
            String newline = new StringBuffer(line).reverse().toString();
            String[] fields = newline.split("\t", 2);
            String[] parts = fields[1].split("@", 2);
            String word = new StringBuilder(parts[1]).reverse().toString();
            String query = conf.get("query").trim();
            // visit all terms in query to check whether is equals to current word
            for (String term : WORD_BOUNDARY.split(query)) {
                if(word.equals(term)) {
                    Text currentKey = new Text(new StringBuilder(parts[0]).reverse().toString());
                    DoubleWritable currentVal = new DoubleWritable(Double.parseDouble(new StringBuilder(fields[0]).reverse().toString()));
                    context.write(currentKey, currentVal);
                    //break;
                }
            }

        }

    }

    public static class SearchReduce extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        //@Override
        public void reduce(Text key,  Iterable<DoubleWritable> values,  Context context)
                throws IOException,  InterruptedException {

            // sum all tf-idf scores
            double sum = 0;
            for (DoubleWritable val : values) {
                sum += val.get();

            }

            context.write(key, new DoubleWritable(sum));
        }
    }

}
