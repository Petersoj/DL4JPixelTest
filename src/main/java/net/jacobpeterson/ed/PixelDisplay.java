package net.jacobpeterson.ed;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;

public class PixelDisplay {

    private JFrame frame;
    private JPanel contentPane;
    private INDArray grayScaleRaster;

    public PixelDisplay() {
        this.frame = new JFrame();
        this.grayScaleRaster = Nd4j.create(new double[4]);
        this.setupWindow();
    }

    private void setupWindow() {
        contentPane = new JPanel() {
            @Override
            public void paint(Graphics g) {
                Graphics2D graphics = (Graphics2D) g;
                Dimension size = getSize();

                int topLeftPixel = (int) (grayScaleRaster.getDouble(0) * 255);
                int topRightPixel = (int) (grayScaleRaster.getDouble(1) * 255);
                int bottomLeftPixel = (int) (grayScaleRaster.getDouble(2) * 255);
                int bottomRightPixel = (int) (grayScaleRaster.getDouble(3) * 255);

                graphics.setColor(new Color(topLeftPixel, topLeftPixel, topLeftPixel));
                graphics.fillRect(0, 0, size.width / 2, size.height / 2);
                graphics.setColor(new Color(topRightPixel, topRightPixel, topRightPixel));
                graphics.fillRect(size.width / 2, 0, size.width / 2, size.height / 2);
                graphics.setColor(new Color(bottomLeftPixel, bottomLeftPixel, bottomLeftPixel));
                graphics.fillRect(0, size.height / 2, size.width / 2, size.height / 2);
                graphics.setColor(new Color(bottomRightPixel, bottomRightPixel, bottomRightPixel));
                graphics.fillRect(size.width / 2, size.height / 2, size.width / 2, size.height / 2);
            }
        };

        frame.setContentPane(contentPane);
        frame.setSize(500, 500);
        frame.setLocationRelativeTo(null); // Centers JFrame
        frame.setVisible(true);
    }

    public void setGrayScaleRaster(INDArray grayScaleRaster) {
        this.grayScaleRaster = grayScaleRaster;
        SwingUtilities.invokeLater(() -> contentPane.repaint());
    }
}
