

import java.sql.*;

public class Main {
    public static void main(String[] args) {
        String url = "jdbc:mysql://localhost:3306/qlsinhvien"; // DB bạn đã tạo
        String user = "root"; // tài khoản
        String password = "hieulatao9"; // Mật khẩu bạn đặt lúc cài

        try (Connection conn = DriverManager.getConnection(url, user, password)) {
            System.out.println("✅ Đã kết nối MySQL");

            // Thực hiện truy vấn
            Statement stmt = conn.createStatement();
            ResultSet rs = stmt.executeQuery("SELECT * FROM sinhvien");

            while (rs.next()) {
                String maSV = rs.getString("maSV");
                String hoTen = rs.getString("hoTen");
                System.out.println(maSV + " - " + hoTen);
            }

        } catch (SQLException e) {
            System.out.println("❌ Kết nối thất bại: " + e.getMessage());
        }
    }
}
