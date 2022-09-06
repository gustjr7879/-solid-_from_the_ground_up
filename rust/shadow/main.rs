fn main() {
    let x = 5;
    let x = x + 1;
    let x = x*2;
    println!("The value of x is : {}",x);
    let spaces = "    ";
    let spaces = spaces.len();
    println!("space size is : {}",spaces);
}
//이처럼 변수에 mut을 안하고 이전 변수를 shadows할 수 있다.
// 
