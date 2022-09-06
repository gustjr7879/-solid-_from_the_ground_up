use std::io; // input output 라이브러리가 std라는 표준에 있다는 뜻
extern crate rand;
use rand::Rng;
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");
    let secret_number = rand::thread_rng().gen_range(1,101);

    println!("The secret number is : {}", secret_number);
    loop{
        println!("Please input your guess.");

        let mut guess = String::new();// 입력값을 저장할 공간
        //let을 사용하여 변수를 선언할 수 있음 
        // let foo = bar; foo라는 변수를 생성하고 bar와 묶는다
        // mut을 붙여야지 가변변수가 되고 mut이 없다면 기본적으로는 불변변수
        // String::new 는 new함수가 새로운 빈 String을 생성한다는 것에 있음
        io::stdin().read_line(&mut guess)
        .expect("Failed to read line");

        let guess: u32 = guess.trim().parse()
            .expect("Please type a number!");
        // 맨 처음 긁어온 io의 연관함수 stdin과 함께 묶고
    // read_line하는데 &라는 참조자를 사용하여 매번 메모리에 복사하지 않고 guess를 가져옴
    // 참조자는 변수와 비슷하게 항상 불변이므로 mut을 사용하여 가변처리해줘야함
    //만약에 expect를 사용하여 처리해주지 않는다면 컴파일은 되지만 경고가 나타남

        println!("You guessed: {}", guess);

        match guess.cmp(&secret_number){
            Ordering::Less => println!("Too small"),
            Ordering::Greater => println!("Too big"),
            Ordering::Equal => {
                println!("Same");
                break;
            }
        }
    }
}
