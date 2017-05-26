package cz.muni.fi.imageNet.core.objects;

/**
 *
 * @author Jakub Peschel
 */
public class Label implements Comparable<Label>{

    private final String labelName;

    public Label(String labelName) {
        this.labelName = labelName;
    }

    public String getLabelName() {
        return labelName;
    }

    @Override
    public boolean equals(Object obj) {
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

    public int compareTo(Label t) {
        return this.getLabelName().compareTo(t.getLabelName());
    }
}
