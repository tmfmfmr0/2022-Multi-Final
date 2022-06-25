import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);
  @override
  Widget build(BuildContext context) {

    // 메인페이지 디자인 - 앱 디자인은 위젯 짜집기 (중요위젯: 글자, 이미지, 아이콘, 박스)
    return MaterialApp(
      home: Text('안녕')
    );

  }
}
