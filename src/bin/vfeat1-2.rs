use netflix_prize::{vfeat1::{Selection, save_vfeat1}, SPLIT_2};

fn main() {
    let sel = Selection::All;
    save_vfeat1("vfeat1", sel, SPLIT_2);
}
