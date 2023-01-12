package nl.andrewl.emaildatasetbrowser.util;

import org.jsoup.Jsoup;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class EmailHelper {

    public static final Pattern replyPattern1 = Pattern.compile("On (Mon|Tue|Wed|Thu|Fri|Sat|Sun), \\d+ " +
            "(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec) \\d+ " +
            "at \\d+:\\d+, .+ wrote:");

    public static final Pattern replyPattern2 = Pattern.compile(
            "On (Mon|Tue|Wed|Thu|Fri|Sat|Sun), " +
                    "(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec) \\d+, " +
                    "\\d+ " +
                    "at \\d+:\\d+ (AM|PM) .+ wrote:"
    );

    private static final Pattern[] patterns = {replyPattern1, replyPattern2};

    /**
     * <p>Some email replies end with a thread of previous emails it is a reply of. This function removes it and replaces
     *      * it with a placeholder, by searching for the following pattern:</p>
     *
     * {@code On (Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d+ (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec) \d+ at \d+:\d+, .+ wrote:}
     *
     * <p>Note that the above pattern assumes dates and times are valid. It is assumed that the formatting of these were
     * done automatically by the email client.</p>
     *
     * @param body A plain-text version of the email body
     *
     * @return In case a mail thread was found, the email body where the thread is replaced with [EBSE: PREVIOUS EMAILS REMOVED].
     * Otherwise, the body remains untouched.
     * */
    public static String replacePrevious(String body) {
        String ret = body;

        // try to match with different formats
        for (Pattern pattern : patterns) {
            Matcher matcher = pattern.matcher(body);
            if (matcher.find()) {
                ret = body.substring(0, matcher.start()) + " [EBSE:PREVIOUS EMAILS REMOVED] ";
                break;
            }
        }

        System.out.printf("""
                    -------- VALIDATION -------
                    Before:
                        %s
                                
                    ----------
                    After:
                        %s
                                
                    ---------------------------
                    %n""", body, ret);

        return ret;
    }

    public static String toPlainText(String html) {
        return Jsoup.parse(html).wholeText();
    }

}
