import javax.swing.*; 
import java.applet.Applet;
import java.awt.*;
import java.io.*;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

/**
 * Legacy AWT Applet client that talks to server.py.
 * Shows text replies and renders images from the Python server.
 * Run with an applet viewer (JDK 8) or CheerpJ Applet Runner.
 */
public class JavaAppletClient extends Applet {
    private static final String HOST = "127.0.0.1";
    private static final int PORT = 1234;
    private static final int SOCKET_TIMEOUT_MS = 20_000;

    private TextArea input;
    private TextArea output;
    private ImageCanvas imageCanvas;

    @Override
    public void init() {
        setLayout(new BorderLayout(6, 6));

        input = new TextArea(3, 60);
        output = new TextArea(10, 60);
        output.setEditable(false);

        Button send = new Button("Send to Python");
        send.addActionListener(e -> onSend());

        Panel top = new Panel(new BorderLayout(6, 6));
        top.add(input, BorderLayout.CENTER);
        top.add(send, BorderLayout.EAST);

        // Image canvas to render plot.jpg/violin.jpg/pair.jpg
        imageCanvas = new ImageCanvas();
        ScrollPane imageScroll = new ScrollPane(ScrollPane.SCROLLBARS_AS_NEEDED);
        imageScroll.add(imageCanvas);

        add(top, BorderLayout.NORTH);
        add(new ScrollPane(ScrollPane.SCROLLBARS_AS_NEEDED) {{ add(output); }}, BorderLayout.CENTER);
        add(imageScroll, BorderLayout.SOUTH);
    }

    private void onSend() {
        String text = input.getText().trim();
        input.setText("");
        if (text.isEmpty()) {
            prepend("Type a command to send to the Python server...");
            return;
        }
        for (String cmd : text.split("\\R")) {
            cmd = cmd.trim();
            if (!cmd.isEmpty()) process(cmd);
        }
    }

    private void process(String cmd) {
        String header = ">>> " + cmd + "\n";
        try (Socket s = new Socket(HOST, PORT)) {
            s.setSoTimeout(SOCKET_TIMEOUT_MS);

            DataInputStream in = new DataInputStream(s.getInputStream());
            DataOutputStream out = new DataOutputStream(s.getOutputStream());

            out.write(cmd.getBytes(StandardCharsets.UTF_8));
            out.flush();

            if ("chart".equals(cmd) || "violin".equals(cmd) || "pair".equals(cmd)) {
                String fileName = "plot.jpg";
                if ("violin".equals(cmd)) fileName = "violin.jpg";
                else if ("pair".equals(cmd)) fileName = "pair.jpg";

                // stream image to disk
                try (FileOutputStream fos = new FileOutputStream(fileName)) {
                    byte[] buf = new byte[8192];
                    int n;
                    while ((n = in.read(buf)) != -1) {
                        fos.write(buf, 0, n);
                    }
                }

                // show in applet
                imageCanvas.setImage(Toolkit.getDefaultToolkit().getImage(new File(fileName).getAbsolutePath()));
                prepend(header + "Visualization displayed below.");
            } else {
                // text reply
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                byte[] buf = new byte[4096];
                int n;
                while ((n = in.read(buf)) != -1) baos.write(buf, 0, n);
                String reply = baos.toString(StandardCharsets.UTF_8.name()).trim();
                imageCanvas.setImage(null); // clear previous image if any
                prepend(header + reply);
            }
        } catch (Exception ex) {
            prepend(header + "Error: " + ex.getClass().getSimpleName() + ": " + ex.getMessage());
        }
    }

    private void prepend(String s) {
        output.setText(s + "\n" + output.getText());
    }

    /** Simple AWT canvas to render a java.awt.Image */
    static class ImageCanvas extends Canvas {
        private Image image;

        void setImage(Image img) {
            this.image = img;
            if (img != null) {
                // optional: scale the canvas to fit a decent height
                setPreferredSize(new Dimension(900, 360));
            }
            repaint();
        }

        @Override
        public void paint(Graphics g) {
            super.paint(g);
            if (image != null) {
                int w = getWidth();
                int h = getHeight();
                // draw centered with scaling to fit
                int imgW = image.getWidth(this);
                int imgH = image.getHeight(this);
                if (imgW > 0 && imgH > 0) {
                    double scale = Math.min((double) w / imgW, (double) h / imgH);
                    int dw = (int) (imgW * scale);
                    int dh = (int) (imgH * scale);
                    int x = (w - dw) / 2;
                    int y = (h - dh) / 2;
                    g.drawImage(image, x, y, dw, dh, this);
                }
            } else {
                g.drawString("No image", 10, 20);
            }
        }
    }
}
