package cz.muni.fi.image.net.core.objects;

/**
 * @author Jakub Peschel (jakubpeschel@gmail.com)
 */
public class Label implements Comparable<Label> {

    private final String labelName;

    public Label(final String labelName) {
        this.labelName = labelName;
    }

    public String getLabelName() {
        return labelName;
    }

    @Override
    public boolean equals(final Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Label other = (Label) obj;
        if ((this.labelName == null) ? (other.labelName != null) : !this.labelName.equals(other.labelName)) {
            return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 67 * hash + (this.labelName != null ? this.labelName.hashCode() : 0);
        return hash;
    }

    public int compareTo(final Label t) {
        return this.getLabelName().compareTo(t.getLabelName());
    }

    @Override
    public String toString() {
        return labelName;
    }
}
