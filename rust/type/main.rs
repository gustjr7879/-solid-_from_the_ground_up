fn main() {
    let x = 2.0; // f64
    let y: f32 = 3.0; // f32지정
    let sum = 5+10;
    let diff = 95.5 - 4.3;
    let product = 4*30;
    let division = 56.7/32.2;
    let remainder = 43%5;
    let t = true;
    let f: bool = false; //boolean false지정
    let c = 'z';
    let z = 'Z';//char
    let tup: (i32,f64,u8) = (500,6.4,1); // tuble 구조화
    let (xx,yy,zz) = tup; // 구조화 된 튜플을 xx yy zz 에 구조해체
    let test: (i32,f64,u8) = (500,6.4,1);
    let five_hundred = test.0;
    let six_point_four = test.1;
    let one =test.2; //튜플을 구조해체할 때 직접적으로 접근하여 구조해체하는 방법
    let a = [1,2,3,4,5];
    let first = a[0];
    let second = a[1];
    println!("{}",x);
    println!("{}",y);
    println!("{}",sum);
    println!("{}",diff);
    println!("{}",product);
    println!("{}",division);
    println!("{}",remainder);
    println!("{}",t);
    println!("{}",f);
    println!("{}",c);
    println!("{}",z);
    //println!("{}",tup);
    println!("{}",xx);
    println!("{}",yy);
    println!("{}",zz);
    println!("{}",five_hundred);
    println!("{}",six_point_four);
    println!("{}",one);
    //println!("{}",a);
    println!("{}",first);
    println!("{}",second);

}
//러스트에서 데이터 타입은 크게 스칼라와 컴파운드로 나뉜다.
//스칼라에는 정수형, 부동소수점 숫자, boolean, 문자 4가지가 있다
//정수형 : 부호가 있으면 i32 없으면 u32와 같이 쓴다
// 8 16 32 64 arch가 있다. 보통 32가 가장 빠름
//부동 소수점 숫자 f32, f64
//boolean True, false
//문자 타입지원 char string, char는 작은따옴표, string은 큰따옴표
//char에는 이모티콘도 사용가능함
//복합타입 : 다른 타입의 다양한 값들을 하나의 타입으로 묶을 수 있음
//tuple 과 배열
// 값들을 집합시켜서 튜플화하기 및 구조해체
// 구조해체를 할 때 순서를 줘서 직접적으로 접근할 수 있다.
//배열은 튜플과 다르게 모든 요소가 같은 타입이여야한다.
//다른 언어와 다른 것은 고정된 길이를 가진다. 즉 한번 선언하면 크기가 고정
//벡터라는 타입은 배열과 다르게 가변적임 러스트에서 배열은 더 변할게 없는 고정된 값일때 씀

//여기서 못한거 : tuple 과 array 출력하기
