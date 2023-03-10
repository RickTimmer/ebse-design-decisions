package nl.andrewl.emaildatasetbrowser.view;

import javax.swing.BorderFactory;
import javax.swing.JCheckBox;
import javax.swing.JDialog;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;

import java.awt.*;
import java.util.prefs.Preferences;

import nl.andrewl.emaildatasetbrowser.EmailDatasetBrowser;
import nl.andrewl.emaildatasetbrowser.view.email.EmailBodyPanel;
import nl.andrewl.emaildatasetbrowser.view.email.TagPanel;
import nl.andrewl.emaildatasetbrowser.view.search.EmailTreeSelectionListener;
import nl.andrewl.emaildatasetbrowser.view.search.searchpanel.SimpleBrowsePanel;

public class SettingsDialog extends JDialog {
    private final Preferences prefs;

    public SettingsDialog(EmailDatasetBrowser browser) {
        super(browser, "Settings");
        this.prefs = EmailDatasetBrowser.getPreferences();

        // TODO: Find a more sophisticated way to set default values for settings.
        // Currently, the implementing code is responsible for this, which might cause
        // inconsistentcies.

        // TODO: Find a more sophisticated way to update objects interested in updated
        // settings. Currently, they're expected to take care of this themselves, which
        // is unreliable.

        JPanel p = new JPanel(new GridLayout(6, 1));
        p.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

        p.add(buildSettingsCheckbox(EmailDatasetBrowser.PREF_LOAD_LAST_DS, false,
                "Open last dataset on start-up"));

        p.add(buildSettingsCheckbox(EmailTreeSelectionListener.PREF_AUTO_OPEN, true,
                "Automatically expand email in tree-view"));

        p.add(buildSettingsSpinner(SimpleBrowsePanel.PREF_BROWSE_PAGE_SIZE, 20, 1, 1000000, 1, "Browse page size"));

        p.add(buildSettingsSpinner(EmailBodyPanel.PREF_SCROLL_SPEED, 100, 1, 200, 1, "Email body scroll speed"));

        p.add(buildSettingsCheckbox(TagPanel.PREF_AUTO_TAG_AUTHOR, false,
                "Automatically add the author tag."));
        p.add(buildSettingsSpinner(TagPanel.PREF_AUTO_TAG_ID, 0, 0, 99999, 1,
                "The tag ID of tag used for the previous setting."));

        setContentPane(p);

        setPreferredSize(new Dimension(475, 275));
        pack();
        setLocationRelativeTo(browser);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
    }

    private JCheckBox buildSettingsCheckbox(String key, boolean defaultValue, String text) {
        JCheckBox openLastDatasetCheckBox = new JCheckBox(text);
        openLastDatasetCheckBox.setSelected(prefs.getBoolean(key, defaultValue));
        openLastDatasetCheckBox
                .addActionListener((e) -> prefs.putBoolean(key, ((JCheckBox) e.getSource()).isSelected()));
        return openLastDatasetCheckBox;
    }

    private LabelledField buildSettingsSpinner(String key, int defaultValue, int min, int max, int stepsize,
            String text) {
        JSpinner spinner = new JSpinner(new SpinnerNumberModel(prefs.getInt(key, defaultValue), min, max, stepsize));
        spinner.addChangeListener((e) -> prefs.putInt(key, (int) spinner.getValue()));
        return new LabelledField(text, spinner);
    }
}
