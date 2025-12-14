import javax.swing.*;
import javax.swing.table.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.*;

public class JavaSwingClient extends JFrame {
    private JTextArea command;
    private JTextArea result;
    private JButton sendButton;
    private JLabel imageShow;
    private JLabel statusLabel;
    private JTextField serverHostField;
    private JTextField serverPortField;
    private JTable dataTable;
    private javax.swing.table.DefaultTableModel tableModel;
    private java.util.List<String> commandHistory;
    private int historyIndex;
    private JComboBox<String> commandComboBox;
    private static final String DEFAULT_HOST = "127.0.0.1";
    private static final int DEFAULT_PORT = 1234;
    
    // Available commands for autocomplete
    private static final String[] AVAILABLE_COMMANDS = {
        "chart", "violin", "pair", "histogram", "boxplot", "heatmap", "barchart", "bar",
        "scattermatrix", "scatter_matrix", "all", "summary", "shape", "head", "tail",
        "columns", "dtypes", "info", "mean", "median", "std", "min", "max", "isnull",
        "nunique", "getdata", "get_data", "filter", "sort", "group"
    };
    
    // Visualization command types
    private static final String[] VISUALIZATION_COMMANDS = {
        "chart", "violin", "pair", "histogram", "boxplot", "heatmap", "barchart", "bar", "scattermatrix", "scatter_matrix"
    };

    public JavaSwingClient() {
        setTitle("Python Data Visualization in Java - Enhanced Swing Client");
        setSize(1200, 800);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout(10, 10));

        // Initialize command history
        commandHistory = new java.util.ArrayList<>();
        historyIndex = -1;

        // Create menu bar
        createMenuBar();
        
        // Top panel with connection settings and file picker
        JPanel topPanel = createTopPanel();
        add(topPanel, BorderLayout.NORTH);

        // Center panel with command input and output
        JPanel centerPanel = createCenterPanel();
        add(centerPanel, BorderLayout.CENTER);

        // Bottom panel with visualization
        JPanel bottomPanel = createBottomPanel();
        add(bottomPanel, BorderLayout.SOUTH);

        // Status bar
        JPanel statusBar = createStatusBar();
        add(statusBar, BorderLayout.PAGE_END);

        setVisible(true);
        updateStatus("Ready - Connect to server to begin", Color.BLACK);
    }

    private void createMenuBar() {
        JMenuBar menuBar = new JMenuBar();
        
        JMenu fileMenu = new JMenu("File");
        JMenuItem loadFileItem = new JMenuItem("Load Dataset...");
        loadFileItem.addActionListener(e -> openFileDialog());
        fileMenu.add(loadFileItem);
        menuBar.add(fileMenu);
        
        JMenu helpMenu = new JMenu("Help");
        JMenuItem aboutItem = new JMenuItem("About");
        aboutItem.addActionListener(e -> showAboutDialog());
        helpMenu.add(aboutItem);
        menuBar.add(helpMenu);
        
        setJMenuBar(menuBar);
    }

    private JPanel createTopPanel() {
        JPanel top = new JPanel(new BorderLayout(5, 5));
        top.setBorder(BorderFactory.createTitledBorder("Connection & Quick Actions"));

        // Connection settings panel
        JPanel connectionPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        connectionPanel.add(new JLabel("Server:"));
        serverHostField = new JTextField(DEFAULT_HOST, 12);
        connectionPanel.add(serverHostField);
        connectionPanel.add(new JLabel("Port:"));
        serverPortField = new JTextField(String.valueOf(DEFAULT_PORT), 6);
        connectionPanel.add(serverPortField);
        
        JButton testConnectionBtn = new JButton("Test Connection");
        testConnectionBtn.addActionListener(e -> testConnection());
        connectionPanel.add(testConnectionBtn);
        
        top.add(connectionPanel, BorderLayout.NORTH);

        // Quick action buttons
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        buttonPanel.add(new JLabel("Quick Actions:"));
        
        // Data operations
        addQuickButton(buttonPanel, "Load File...", e -> {
            openFileDialog();
            // Auto-load table after a short delay
            new Thread(() -> {
                try { Thread.sleep(500); } catch (InterruptedException ex) {}
                loadDataTable();
            }).start();
        });
        addQuickButton(buttonPanel, "View Data", e -> loadDataTable());
        addQuickButton(buttonPanel, "Summary", e -> sendCommand("summary"));
        addQuickButton(buttonPanel, "Shape", e -> sendCommand("shape"));
        addQuickButton(buttonPanel, "Head", e -> sendCommand("head"));
        
        buttonPanel.add(Box.createHorizontalStrut(20));
        buttonPanel.add(new JLabel("Visualizations:"));
        
        // Visualization buttons
        addQuickButton(buttonPanel, "Regression", e -> sendCommand("chart"));
        addQuickButton(buttonPanel, "Violin", e -> sendCommand("violin"));
        addQuickButton(buttonPanel, "Pairplot", e -> sendCommand("pair"));
        addQuickButton(buttonPanel, "Histogram", e -> sendCommand("histogram"));
        addQuickButton(buttonPanel, "Boxplot", e -> sendCommand("boxplot"));
        addQuickButton(buttonPanel, "Heatmap", e -> sendCommand("heatmap"));
        addQuickButton(buttonPanel, "Bar Chart", e -> sendCommand("barchart"));
        addQuickButton(buttonPanel, "Scatter Matrix", e -> sendCommand("scattermatrix"));
        addQuickButton(buttonPanel, "All Charts", e -> sendCommand("all"));
        
        top.add(buttonPanel, BorderLayout.CENTER);
        return top;
    }

    private void addQuickButton(JPanel panel, String text, ActionListener listener) {
        JButton btn = new JButton(text);
        btn.setFont(new Font("Arial", Font.PLAIN, 11));
        btn.addActionListener(listener);
        panel.add(btn);
    }

    private JPanel createCenterPanel() {
        JPanel center = new JPanel(new BorderLayout(5, 5));
        
        // Command input panel
        JPanel commandPanel = new JPanel(new BorderLayout(5, 5));
        commandPanel.setBorder(BorderFactory.createTitledBorder("Input Command to Python Server"));
        
        // Command history combo box
        commandComboBox = new JComboBox<>();
        commandComboBox.setEditable(true);
        commandComboBox.setFont(new Font("Monospaced", Font.PLAIN, 12));
        JTextField comboTextField = (JTextField) commandComboBox.getEditor().getEditorComponent();
        comboTextField.addKeyListener(new CommandAutocompleteListener(comboTextField));
        commandComboBox.addActionListener(e -> {
            if (commandComboBox.getSelectedItem() != null) {
                String selected = commandComboBox.getSelectedItem().toString();
                if (!selected.isEmpty() && !commandHistory.contains(selected)) {
                    commandHistory.add(0, selected);
                    updateCommandComboBox();
                }
            }
        });
        
        // Command text area (alternative input)
        command = new JTextArea(3, 50);
        command.setFont(new Font("Monospaced", Font.PLAIN, 12));
        command.setLineWrap(true);
        command.setWrapStyleWord(true);
        command.addKeyListener(new CommandHistoryListener());
        
        // Panel for both input methods
        JPanel inputPanel = new JPanel(new BorderLayout(3, 3));
        inputPanel.add(new JLabel("Command History:"), BorderLayout.NORTH);
        inputPanel.add(commandComboBox, BorderLayout.CENTER);
        inputPanel.add(new JLabel("Or type command below:"), BorderLayout.SOUTH);
        
        JPanel commandInputPanel = new JPanel(new BorderLayout(3, 3));
        commandInputPanel.add(inputPanel, BorderLayout.NORTH);
        commandInputPanel.add(new JScrollPane(command), BorderLayout.CENTER);
        
        commandPanel.add(commandInputPanel, BorderLayout.CENTER);
        
        sendButton = new JButton("Send to Python");
        sendButton.setFont(new Font("Arial", Font.BOLD, 14));
        sendButton.addActionListener(new SendButtonListener());
        commandPanel.add(sendButton, BorderLayout.EAST);
        
        // Create tabbed pane for Output and Data Table
        JTabbedPane tabbedPane = new JTabbedPane();
        
        // Result output panel
        JPanel resultPanel = new JPanel(new BorderLayout(5, 5));
        result = new JTextArea(12, 50);
        result.setFont(new Font("Monospaced", Font.PLAIN, 12));
        result.setEditable(false);
        result.setBackground(new Color(245, 245, 245));
        
        JScrollPane resultScroll = new JScrollPane(result);
        resultScroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        resultPanel.add(resultScroll, BorderLayout.CENTER);
        
        // Clear button
        JButton clearButton = new JButton("Clear Output");
        clearButton.addActionListener(e -> result.setText(""));
        resultPanel.add(clearButton, BorderLayout.SOUTH);
        
        // Data table panel
        JPanel tablePanel = new JPanel(new BorderLayout(5, 5));
        tableModel = new DefaultTableModel();
        dataTable = new JTable(tableModel);
        dataTable.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
        dataTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        dataTable.setFont(new Font("Monospaced", Font.PLAIN, 11));
        
        JScrollPane tableScroll = new JScrollPane(dataTable);
        tableScroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        tableScroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
        tablePanel.add(tableScroll, BorderLayout.CENTER);
        
        // Table controls
        JPanel tableControls = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton refreshTableBtn = new JButton("Refresh Table");
        refreshTableBtn.addActionListener(e -> loadDataTable());
        JButton exportDataBtn = new JButton("Export to CSV...");
        exportDataBtn.addActionListener(e -> exportTableToCSV());
        JButton clearTableBtn = new JButton("Clear Table");
        clearTableBtn.addActionListener(e -> {
            tableModel.setRowCount(0);
            tableModel.setColumnCount(0);
        });
        tableControls.add(refreshTableBtn);
        tableControls.add(exportDataBtn);
        tableControls.add(clearTableBtn);
        tablePanel.add(tableControls, BorderLayout.SOUTH);
        
        tabbedPane.addTab("Output", resultPanel);
        tabbedPane.addTab("Data Table", tablePanel);
        
        center.add(commandPanel, BorderLayout.NORTH);
        center.add(tabbedPane, BorderLayout.CENTER);
        return center;
    }

    private JPanel createBottomPanel() {
        JPanel bot = new JPanel(new BorderLayout(5, 5));
        bot.setBorder(BorderFactory.createTitledBorder("Visualization from Python"));
        bot.setPreferredSize(new Dimension(1200, 350));
        imageShow = new JLabel("No visualization loaded", SwingConstants.CENTER);
        imageShow.setOpaque(true);
        imageShow.setBackground(Color.WHITE);
        imageShow.setForeground(Color.GRAY);
        imageShow.setFont(new Font("Arial", Font.ITALIC, 14));
        
        JScrollPane imageScroll = new JScrollPane(imageShow);
        imageScroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
        imageScroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        bot.add(imageScroll, BorderLayout.CENTER);
        
        // Image controls
        JPanel imageControls = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        JButton clearImageBtn = new JButton("Clear Image");
        clearImageBtn.addActionListener(e -> {
            imageShow.setIcon(null);
            imageShow.setText("No visualization loaded");
        });
        JButton exportImageBtn = new JButton("Save Image...");
        exportImageBtn.addActionListener(e -> exportCurrentImage());
        imageControls.add(exportImageBtn);
        imageControls.add(clearImageBtn);
        bot.add(imageControls, BorderLayout.SOUTH);
        
        return bot;
    }

    private JPanel createStatusBar() {
        JPanel statusBar = new JPanel(new BorderLayout());
        statusBar.setBorder(BorderFactory.createLoweredBevelBorder());
        statusLabel = new JLabel("Ready");
        statusLabel.setBorder(BorderFactory.createEmptyBorder(3, 5, 3, 5));
        statusBar.add(statusLabel, BorderLayout.WEST);
        return statusBar;
    }

    private void updateStatus(String message, Color color) {
        statusLabel.setText(message);
        statusLabel.setForeground(color);
    }

    private void openFileDialog() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
            @Override
            public boolean accept(File f) {
                return f.isDirectory() || f.getName().toLowerCase().endsWith(".csv") 
                    || f.getName().toLowerCase().endsWith(".txt");
            }
            
            @Override
            public String getDescription() {
                return "CSV/TXT Files (*.csv, *.txt)";
            }
        });
        
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            String filePath = selectedFile.getAbsolutePath();
            sendCommand(filePath);
        }
    }

    private void testConnection() {
        SwingUtilities.invokeLater(() -> {
            updateStatus("Testing connection...", Color.BLUE);
            new Thread(() -> {
                try {
                    String host = serverHostField.getText().trim();
                    int port = Integer.parseInt(serverPortField.getText().trim());
                    try (Socket s = new Socket()) {
                        s.connect(new InetSocketAddress(host, port), 2000);
                        SwingUtilities.invokeLater(() -> {
                            updateStatus("Connection successful!", Color.GREEN);
                            prependResult(">>> Connection test successful\n");
                        });
                    }
                } catch (Exception ex) {
                    SwingUtilities.invokeLater(() -> {
                        updateStatus("Connection failed: " + ex.getMessage(), Color.RED);
                        prependResult(">>> Connection test failed: " + ex.getMessage() + "\n");
                    });
                }
            }).start();
        });
    }

    private void sendCommand(String cmd) {
        command.setText(cmd);
        sendButton.doClick();
    }

    private void updateCommandComboBox() {
        commandComboBox.removeAllItems();
        for (String cmd : commandHistory) {
            commandComboBox.addItem(cmd);
        }
    }

    private void addToHistory(String cmd) {
        if (cmd != null && !cmd.trim().isEmpty()) {
            commandHistory.remove(cmd); // Remove if exists
            commandHistory.add(0, cmd); // Add to beginning
            if (commandHistory.size() > 50) { // Limit to 50 commands
                commandHistory.remove(commandHistory.size() - 1);
            }
            updateCommandComboBox();
            historyIndex = -1;
        }
    }

    class CommandHistoryListener extends KeyAdapter {
        @Override
        public void keyPressed(KeyEvent e) {
            if (e.getKeyCode() == KeyEvent.VK_UP) {
                e.consume();
                if (historyIndex < commandHistory.size() - 1) {
                    historyIndex++;
                    command.setText(commandHistory.get(historyIndex));
                }
            } else if (e.getKeyCode() == KeyEvent.VK_DOWN) {
                e.consume();
                if (historyIndex > 0) {
                    historyIndex--;
                    command.setText(commandHistory.get(historyIndex));
                } else if (historyIndex == 0) {
                    historyIndex = -1;
                    command.setText("");
                }
            } else {
                historyIndex = -1;
            }
        }
    }

    class CommandAutocompleteListener extends KeyAdapter {
        private JTextField textField;
        private JPopupMenu popupMenu;
        private java.util.List<String> suggestions;

        public CommandAutocompleteListener(JTextField field) {
            this.textField = field;
            this.popupMenu = new JPopupMenu();
            this.suggestions = new java.util.ArrayList<>();
        }

        @Override
        public void keyReleased(KeyEvent e) {
            if (e.getKeyCode() == KeyEvent.VK_ENTER || e.getKeyCode() == KeyEvent.VK_ESCAPE) {
                popupMenu.setVisible(false);
                return;
            }
            
            String text = textField.getText();
            if (text.length() < 1) {
                popupMenu.setVisible(false);
                return;
            }

            // Find matching commands
            suggestions.clear();
            String lowerText = text.toLowerCase();
            for (String cmd : AVAILABLE_COMMANDS) {
                if (cmd.toLowerCase().startsWith(lowerText)) {
                    suggestions.add(cmd);
                }
            }
            
            // Also check history
            for (String cmd : commandHistory) {
                if (cmd.toLowerCase().startsWith(lowerText) && !suggestions.contains(cmd)) {
                    suggestions.add(cmd);
                }
            }

            if (suggestions.isEmpty()) {
                popupMenu.setVisible(false);
                return;
            }

            // Show suggestions
            popupMenu.removeAll();
            for (String suggestion : suggestions) {
                JMenuItem item = new JMenuItem(suggestion);
                item.addActionListener(ev -> {
                    textField.setText(suggestion);
                    popupMenu.setVisible(false);
                });
                popupMenu.add(item);
            }

            // Position popup below text field
            if (!popupMenu.isVisible()) {
                popupMenu.show(textField, 0, textField.getHeight());
            }
        }
    }

    class SendButtonListener implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            String sendData = command.getText().trim();
            if (sendData.isEmpty()) {
                // Try combo box
                Object selected = commandComboBox.getSelectedItem();
                if (selected != null) {
                    sendData = selected.toString().trim();
                }
            }
            
            if (sendData.isEmpty()) {
                prependResult(">>> Type the command to send to the Python server...\n");
                return;
            }
            
            // Add to history
            addToHistory(sendData);
            command.setText("");
            commandComboBox.setSelectedItem("");
            
            // Disable button during processing
            sendButton.setEnabled(false);
            updateStatus("Processing command...", Color.BLUE);
            
            // Process in background thread
            new Thread(() -> {
                for (String xcode : sendData.split("\n")) {
                    process(xcode.trim());
                }
                SwingUtilities.invokeLater(() -> {
                    sendButton.setEnabled(true);
                    updateStatus("Ready", Color.BLACK);
                });
            }).start();
        }

        private void process(String xcode) {
            String header = ">>> " + xcode + "\n";
            String host = serverHostField.getText().trim();
            int port;
            try {
                port = Integer.parseInt(serverPortField.getText().trim());
            } catch (NumberFormatException e) {
                SwingUtilities.invokeLater(() -> {
                    prependResult(header + "Error: Invalid port number\n");
                    updateStatus("Error: Invalid port", Color.RED);
                });
                return;
            }
            
            try (Socket s = new Socket(host, port);
                 DataInputStream in = new DataInputStream(s.getInputStream());
                 DataOutputStream out = new DataOutputStream(s.getOutputStream())) {
                
                s.setSoTimeout(60000); // 60 second timeout

                out.write(xcode.getBytes("UTF-8"));
                out.flush();

                // Check if it's a visualization command
                boolean isVisualization = false;
                String fileName = null;
                for (String vizCmd : VISUALIZATION_COMMANDS) {
                    if (xcode.equals(vizCmd)) {
                        isVisualization = true;
                        fileName = getVisualizationFileName(vizCmd);
                        break;
                    }
                }

                if (isVisualization && fileName != null) {
                    // Handle image response - use absolute path
                    final String finalFileName = fileName; // Make final for lambda
                    File f = new File(finalFileName);
                    String absolutePath = f.getAbsolutePath();
                    File absoluteFile = new File(absolutePath);
                    
                    if (absoluteFile.exists()) absoluteFile.delete();
                    
                    try (FileOutputStream fos = new FileOutputStream(absoluteFile)) {
                        byte[] buf = new byte[8192];
                        int n;
                        while ((n = in.read(buf)) != -1) {
                            fos.write(buf, 0, n);
                        }
                        fos.flush();
                    }
                    
                    // Small delay to ensure file is fully written
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    
                    // Update UI on EDT
                    final String finalAbsolutePath = absolutePath; // Final for lambda
                    SwingUtilities.invokeLater(() -> {
                        File imageFile = new File(finalAbsolutePath);
                        if (!imageFile.exists()) {
                            prependResult(header + "Error: Image file not found: " + finalAbsolutePath + "\n");
                            updateStatus("Error: Image not found", Color.RED);
                            return;
                        }
                        
                        try {
                            // Load image with absolute path
                            ImageIcon icon = new ImageIcon(finalAbsolutePath);
                            
                            // Wait for image to load
                            if (icon.getImageLoadStatus() != MediaTracker.COMPLETE) {
                                MediaTracker tracker = new MediaTracker(imageShow);
                                tracker.addImage(icon.getImage(), 0);
                                try {
                                    tracker.waitForID(0, 2000);
                                } catch (InterruptedException e) {
                                    Thread.currentThread().interrupt();
                                }
                            }
                            
                            if (icon.getIconWidth() <= 0 || icon.getIconHeight() <= 0) {
                                prependResult(header + "Error: Invalid image file or image not loaded\n");
                                updateStatus("Error: Invalid image", Color.RED);
                                return;
                            }
                            
                            // Scale image if too large
                            Image img = icon.getImage();
                            int maxWidth = 1000;
                            int maxHeight = 600;
                            int imgWidth = icon.getIconWidth();
                            int imgHeight = icon.getIconHeight();
                            
                            double scale = Math.min((double)maxWidth / imgWidth, (double)maxHeight / imgHeight);
                            if (scale > 1.0) scale = 1.0; // Don't upscale
                            
                            int scaledWidth = (int)(imgWidth * scale);
                            int scaledHeight = (int)(imgHeight * scale);
                            
                            Image scaledImg = img.getScaledInstance(
                                scaledWidth,
                                scaledHeight,
                                Image.SCALE_SMOOTH
                            );
                            
                            // Create new icon from scaled image
                            ImageIcon scaledIcon = new ImageIcon(scaledImg);
                            imageShow.setIcon(scaledIcon);
                            imageShow.setText("");
                            
                            // Force repaint
                            imageShow.revalidate();
                            imageShow.repaint();
                            
                            prependResult(header + "Visualization displayed below. File: " + finalAbsolutePath + "\n");
                            updateStatus("Visualization loaded: " + finalFileName, Color.GREEN);
                        } catch (Exception ex) {
                            prependResult(header + "Error loading image: " + ex.getMessage() + "\n");
                            updateStatus("Error loading image: " + ex.getMessage(), Color.RED);
                            ex.printStackTrace();
                        }
                    });
                } else {
                    // Handle text response
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    byte[] buf = new byte[4096];
                    int n;
                    while ((n = in.read(buf)) != -1) {
                        baos.write(buf, 0, n);
                    }
                    String response = baos.toString("UTF-8").trim();
                    
                    SwingUtilities.invokeLater(() -> {
                        prependResult(header + response + "\n");
                        if (response.startsWith("Error")) {
                            updateStatus("Command error", Color.RED);
                        } else {
                            updateStatus("Command executed successfully", Color.GREEN);
                        }
                    });
                }
            } catch (SocketTimeoutException ex) {
                SwingUtilities.invokeLater(() -> {
                    prependResult(header + "Error: Connection timeout\n");
                    updateStatus("Connection timeout", Color.RED);
                });
            } catch (ConnectException ex) {
                SwingUtilities.invokeLater(() -> {
                    prependResult(header + "Error: Could not connect to server. Make sure the server is running.\n");
                    updateStatus("Connection failed", Color.RED);
                });
            } catch (Exception ex) {
                SwingUtilities.invokeLater(() -> {
                    prependResult(header + "Error: " + ex.getClass().getSimpleName() + ": " + ex.getMessage() + "\n");
                    updateStatus("Error: " + ex.getMessage(), Color.RED);
                });
            }
        }

        private String getVisualizationFileName(String cmd) {
            switch (cmd) {
                case "chart": return "plot.jpg";
                case "violin": return "violin.jpg";
                case "pair": return "pair.jpg";
                case "histogram": return "histogram.jpg";
                case "boxplot": return "boxplot.jpg";
                case "heatmap": return "heatmap.jpg";
                case "barchart":
                case "bar": return "barchart.jpg";
                case "scattermatrix":
                case "scatter_matrix": return "scattermatrix.jpg";
                default: return "plot.jpg";
            }
        }

        private void prependResult(String text) {
            SwingUtilities.invokeLater(() -> {
                result.setText(text + result.getText());
                result.setCaretPosition(0);
            });
        }
    }

    private void prependResult(String text) {
        result.setText(text + result.getText());
        result.setCaretPosition(0);
    }

    private void loadDataTable() {
        new Thread(() -> {
            String host = serverHostField.getText().trim();
            int port;
            try {
                port = Integer.parseInt(serverPortField.getText().trim());
            } catch (NumberFormatException e) {
                SwingUtilities.invokeLater(() -> {
                    prependResult(">>> Error: Invalid port number\n");
                    updateStatus("Error: Invalid port", Color.RED);
                });
                return;
            }
            
            try (Socket s = new Socket(host, port);
                 DataInputStream in = new DataInputStream(s.getInputStream());
                 DataOutputStream out = new DataOutputStream(s.getOutputStream())) {
                
                s.setSoTimeout(60000);
                out.write("getdata".getBytes("UTF-8"));
                out.flush();
                
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                byte[] buf = new byte[4096];
                int n;
                while ((n = in.read(buf)) != -1) {
                    baos.write(buf, 0, n);
                }
                
                String response = baos.toString("UTF-8").trim();
                
                if (response.startsWith("Error")) {
                    SwingUtilities.invokeLater(() -> {
                        prependResult(">>> " + response + "\n");
                        updateStatus("Error loading data", Color.RED);
                    });
                    return;
                }
                
                // Parse JSON response (simple parser)
                parseAndDisplayData(response);
                
            } catch (Exception ex) {
                SwingUtilities.invokeLater(() -> {
                    prependResult(">>> Error loading data table: " + ex.getMessage() + "\n");
                    updateStatus("Error loading data", Color.RED);
                });
            }
        }).start();
    }

    private void parseAndDisplayData(String jsonStr) {
        try {
            // Simple JSON parser for our specific format
            // Format: {"columns": [...], "data": [[...], [...]], "shape": [rows, cols]}
            jsonStr = jsonStr.trim();
            if (!jsonStr.startsWith("{") || !jsonStr.endsWith("}")) {
                throw new Exception("Invalid JSON format");
            }
            
            // Extract columns
            int colsStart = jsonStr.indexOf("\"columns\":[") + 11;
            int colsEnd = jsonStr.indexOf("]", colsStart);
            String colsStr = jsonStr.substring(colsStart, colsEnd);
            String[] columns = parseStringArray(colsStr);
            
            // Extract data
            int dataStart = jsonStr.indexOf("\"data\":[") + 8;
            int dataEnd = jsonStr.lastIndexOf("]");
            String dataStr = jsonStr.substring(dataStart, dataEnd);
            String[][] data = parseDataArray(dataStr, columns.length);
            
            // Update table on EDT
            SwingUtilities.invokeLater(() -> {
                tableModel.setColumnCount(0);
                tableModel.setRowCount(0);
                
                // Set column names
                for (String col : columns) {
                    tableModel.addColumn(col);
                }
                
                // Add rows
                for (String[] row : data) {
                    tableModel.addRow(row);
                }
                
                updateStatus("Data table loaded: " + data.length + " rows", Color.GREEN);
                prependResult(">>> Data table refreshed with " + data.length + " rows\n");
            });
            
        } catch (Exception e) {
            SwingUtilities.invokeLater(() -> {
                prependResult(">>> Error parsing data: " + e.getMessage() + "\n");
                updateStatus("Error parsing data", Color.RED);
            });
        }
    }

    private String[] parseStringArray(String str) {
        str = str.trim();
        if (str.isEmpty()) return new String[0];
        
        java.util.List<String> list = new java.util.ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        
        for (char c : str.toCharArray()) {
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                String val = current.toString().trim().replace("\"", "");
                if (!val.isEmpty()) list.add(val);
                current = new StringBuilder();
            } else {
                current.append(c);
            }
        }
        String val = current.toString().trim().replace("\"", "");
        if (!val.isEmpty()) list.add(val);
        
        return list.toArray(new String[0]);
    }

    private String[][] parseDataArray(String str, int numCols) {
        str = str.trim();
        if (str.isEmpty()) return new String[0][];
        
        java.util.List<String[]> rows = new java.util.ArrayList<>();
        StringBuilder current = new StringBuilder();
        int bracketDepth = 0;
        boolean inQuotes = false;
        
        for (char c : str.toCharArray()) {
            if (c == '"') {
                inQuotes = !inQuotes;
                current.append(c);
            } else if (c == '[' && !inQuotes) {
                bracketDepth++;
                if (bracketDepth == 1) {
                    current = new StringBuilder();
                    continue;
                }
                current.append(c);
            } else if (c == ']' && !inQuotes) {
                bracketDepth--;
                if (bracketDepth == 0) {
                    String rowStr = current.toString().trim();
                    if (!rowStr.isEmpty()) {
                        String[] row = parseStringArray(rowStr);
                        if (row.length == numCols) {
                            rows.add(row);
                        }
                    }
                    current = new StringBuilder();
                    continue;
                }
                current.append(c);
            } else if (c == ',' && bracketDepth == 1 && !inQuotes) {
                // Skip outer commas
            } else {
                current.append(c);
            }
        }
        
        return rows.toArray(new String[0][]);
    }

    private void exportCurrentImage() {
        Icon icon = imageShow.getIcon();
        if (icon == null) {
            JOptionPane.showMessageDialog(this, "No image to export.", "Export Error", JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
            @Override
            public boolean accept(File f) {
                return f.isDirectory() || f.getName().toLowerCase().endsWith(".jpg") 
                    || f.getName().toLowerCase().endsWith(".png");
            }
            
            @Override
            public String getDescription() {
                return "Image Files (*.jpg, *.png)";
            }
        });
        fileChooser.setDialogTitle("Save Image As");
        
        int result = fileChooser.showSaveDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            String filePath = file.getAbsolutePath();
            if (!filePath.toLowerCase().endsWith(".jpg") && !filePath.toLowerCase().endsWith(".png")) {
                filePath += ".jpg";
            }
            
            try {
                if (icon instanceof ImageIcon) {
                    ImageIcon imgIcon = (ImageIcon) icon;
                    Image img = imgIcon.getImage();
                    java.awt.image.BufferedImage bufferedImage = new java.awt.image.BufferedImage(
                        img.getWidth(null), img.getHeight(null), java.awt.image.BufferedImage.TYPE_INT_RGB
                    );
                    Graphics2D g2d = bufferedImage.createGraphics();
                    g2d.drawImage(img, 0, 0, null);
                    g2d.dispose();
                    
                    javax.imageio.ImageIO.write(bufferedImage, 
                        filePath.toLowerCase().endsWith(".png") ? "PNG" : "JPEG", 
                        new File(filePath));
                    
                    JOptionPane.showMessageDialog(this, "Image saved successfully!", "Export Success", 
                        JOptionPane.INFORMATION_MESSAGE);
                    updateStatus("Image exported: " + filePath, Color.GREEN);
                }
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Error saving image: " + ex.getMessage(), 
                    "Export Error", JOptionPane.ERROR_MESSAGE);
                updateStatus("Error exporting image", Color.RED);
            }
        }
    }

    private void exportTableToCSV() {
        if (tableModel.getRowCount() == 0) {
            JOptionPane.showMessageDialog(this, "No data to export.", "Export Error", JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
            @Override
            public boolean accept(File f) {
                return f.isDirectory() || f.getName().toLowerCase().endsWith(".csv");
            }
            
            @Override
            public String getDescription() {
                return "CSV Files (*.csv)";
            }
        });
        fileChooser.setDialogTitle("Export Data to CSV");
        
        int result = fileChooser.showSaveDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            String filePath = file.getAbsolutePath();
            if (!filePath.toLowerCase().endsWith(".csv")) {
                filePath += ".csv";
            }
            
            try (PrintWriter pw = new PrintWriter(new FileWriter(filePath))) {
                // Write header
                int colCount = tableModel.getColumnCount();
                for (int i = 0; i < colCount; i++) {
                    pw.print(tableModel.getColumnName(i));
                    if (i < colCount - 1) pw.print(",");
                }
                pw.println();
                
                // Write data
                int rowCount = tableModel.getRowCount();
                for (int i = 0; i < rowCount; i++) {
                    for (int j = 0; j < colCount; j++) {
                        Object value = tableModel.getValueAt(i, j);
                        String str = (value == null) ? "" : value.toString();
                        // Escape commas and quotes
                        if (str.contains(",") || str.contains("\"") || str.contains("\n")) {
                            str = "\"" + str.replace("\"", "\"\"") + "\"";
                        }
                        pw.print(str);
                        if (j < colCount - 1) pw.print(",");
                    }
                    pw.println();
                }
                
                JOptionPane.showMessageDialog(this, "Data exported successfully!\nRows: " + rowCount, 
                    "Export Success", JOptionPane.INFORMATION_MESSAGE);
                updateStatus("Data exported: " + filePath + " (" + rowCount + " rows)", Color.GREEN);
                prependResult(">>> Data exported to: " + filePath + "\n");
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Error exporting data: " + ex.getMessage(), 
                    "Export Error", JOptionPane.ERROR_MESSAGE);
                updateStatus("Error exporting data", Color.RED);
            }
        }
    }

    private void showAboutDialog() {
        String aboutText = "Python Data Visualization Server Client\n\n" +
                          "Version: 2.1 (Enhanced)\n" +
                          "Supports multiple visualization types:\n" +
                          "- Regression Charts\n" +
                          "- Violin Plots\n" +
                          "- Pairplots\n" +
                          "- Histograms\n" +
                          "- Boxplots\n" +
                          "- Correlation Heatmaps\n\n" +
                          "Features:\n" +
                          "- File picker for dataset loading\n" +
                          "- Quick action buttons\n" +
                          "- Data table view\n" +
                          "- Export charts and data\n" +
                          "- Connection status indicator\n" +
                          "- Enhanced error handling";
        JOptionPane.showMessageDialog(this, aboutText, "About", JOptionPane.INFORMATION_MESSAGE);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception e) {
                // Use default look and feel
            }
            new JavaSwingClient();
        });
    }
}
