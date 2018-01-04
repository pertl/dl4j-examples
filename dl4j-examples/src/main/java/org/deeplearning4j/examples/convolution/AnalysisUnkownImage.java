package org.deeplearning4j.examples.convolution;

import javax.swing.*;
import javax.swing.border.LineBorder;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.io.File;
import java.util.Arrays;
import java.util.Objects;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

final class AnalysisUnkownImage extends JFrame {

    private static final Logger LOG = LoggerFactory.getLogger(AnalysisUnkownImage.class);

    private final JTextArea jta;
    private final MultiLayerNetwork model;

    AnalysisUnkownImage(String filename, MultiLayerNetwork model, int width, int height) {
        this.model = model;
        JLabel label1 = new JLabel("Directory/File(Plz fill in path");
        JLabel label3 = new JLabel("  skymind.ai");
        JLabel label4 = new JLabel("welcome to here  ");
        label3.setBackground(new Color(87, 105, 227));
        label4.setBackground(new Color(87, 105, 227));
        label3.setForeground(new Color(28, 11, 185));
        label4.setForeground(new Color(239, 16, 228));
        JButton jbt = new JButton("The identification of unknown and show(Click Here)");
        JTextField jtf = new JTextField(380);
        jtf.setText(filename);
        jta = new JTextArea(15, 80);
        jta.setLineWrap(true);
        jta.setWrapStyleWord(true);
        JScrollPane jsp = new JScrollPane(jta);
        LineBorder lb = new LineBorder(Color.red, 1);
        TitledBorder tb = new TitledBorder(new TitledBorder(""), " Identification result", TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.BOTTOM, new Font("Serif", Font.BOLD, 12), Color.BLUE);

        JPanel panel = new JPanel();
        JPanel panel1 = new JPanel();
        JPanel panel2 = new JPanel();
        JPanel panel3 = new JPanel();
        JPanel panel4 = new JPanel();
        JPanel panel5 = new JPanel();
        JPanel panel7 = new JPanel();

        panel5.setLayout(new BorderLayout(5, 5));
        panel5.add(label1, BorderLayout.WEST);
        panel5.add(jtf, BorderLayout.CENTER);
        panel7.setLayout(new GridLayout(1, 2, 5, 5));
        panel7.add(panel5);
        panel1.setLayout(new BorderLayout(5, 5));
        panel1.setBorder(lb);
        panel1.add(panel7, BorderLayout.CENTER);
        jbt.setForeground(Color.BLUE);
        jbt.setBackground(Color.GREEN);
        panel2.setLayout(new BorderLayout(5, 5));
        panel2.add(label3, BorderLayout.WEST);
        panel2.add(jbt, BorderLayout.CENTER);
        panel2.add(label4, BorderLayout.EAST);
        panel3.setLayout(new BorderLayout(1, 1));
        panel3.add(jsp, BorderLayout.CENTER);
        panel3.setBorder(tb);
        panel4.setLayout(new BorderLayout(5, 5));
        panel4.add(panel2, BorderLayout.NORTH);
        panel4.add(panel3, BorderLayout.CENTER);

        panel.setLayout(new BorderLayout(5, 5));
        panel.add(panel1, BorderLayout.NORTH);
        panel.add(panel4, BorderLayout.CENTER);
        //register listener
        jbt.addActionListener(ae -> {
            String fileAbsolutePath = jtf.getText();
            jta.setText("labels->->->");
            jta.append("\n");
            File file = new File(fileAbsolutePath);
            File[] files = new File[1];
            if (file.exists()) {
                if (file.isDirectory()) {
                    files = file.listFiles();
                } else {
                    files[0] = file;
                }
                assert files != null;
                analysisFileName(files, width, height);

            }
        });
        add(panel);
    }

    private void analysisFileName(File[] files, int width, int height) {
        for (final File file : files) {
            if (file.isDirectory()) {
                analysisFileName(Objects.requireNonNull(file.listFiles()), width, height);
            } else {
                //the suffix of the file
                String suffix = file.getName();
                suffix = suffix.substring(suffix.lastIndexOf(".") + 1);
                String formatAllows = StringUtils.arrayToString(NativeImageLoader.ALLOWED_FORMATS);
                if (formatAllows.contains(suffix)) {
                    // Use NativeImageLoader to convert to numerical matrix
                    NativeImageLoader loader = new NativeImageLoader(height, width, 3);
                    // Get the image into an INDarray
                    INDArray image = null;
                    try {
                        image = loader.asMatrix(file);
                    } catch (Exception e) {
                        LOG.error("the loader.asMatrix have any abnormal", e);
                    }
                    if (image == null) {
                        return;
                    }
                   /* DataNormalization scaler = new ImagePreProcessingScaler(0,1);
                    scaler.transform(image);*/
                    INDArray output = model.output(image);

                    LOG.info("## The Neural Nets Pediction ##");
                    LOG.info("## list of probabilities per label ##");
                    //log.info("## List of Labels in Order## ");
                    // In new versions labels are always in order
                    LOG.info(output.toString());

                    String modelResult = output.toString();

                    int[] predict = model.predict(image);
                    modelResult += "===" + Arrays.toString(predict);
                    jta.append("the file chosen:");
                    jta.append("\n");
                    jta.append(file.getAbsolutePath());
                    jta.append("\n");
                    jta.append("the  identification result :" + modelResult);
                    jta.append("\n");

                }

            }
        }
    }

    void showGUI() {
        setSize(560, 500);
        setLocationRelativeTo(null);
        setBackground(Color.green);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setVisible(true);
    }
}
